#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["rich"]
# ///
"""
TUI dashboard for atack — All-to-All CUDA Kubernetes test.

Displays three panels:
  - Pods: live status of atack StatefulSet pods
  - ComputeDomain daemons: status of computedomain-daemon pods
  - Bandwidth matrix: NVLink bandwidth (GB/s) between all pod pairs

Uses rich for terminal rendering. Manages kubectl logs -f subprocesses
per pod to collect bandwidth measurements. Press Ctrl-C to quit.
"""

import datetime
import fcntl
import json
import logging
import os
import re
import select
import subprocess
import sys
import termios
import threading
import time
import traceback
import tty
import urllib.error
import urllib.request


# Dashboard diagnostics go to a log file to avoid flickering caused by
# stderr writes bleeding through the TUI's alternate screen buffer.
# The log file path is printed to stderr on startup so the user knows
# where to look.
_dashboard_log_path = "/tmp/atack-dashboard.log"
# Force immediate flush so crash diagnostics aren't lost in the buffer.
_log_handler = logging.FileHandler(_dashboard_log_path, mode="w")
_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
_log_handler.stream.reconfigure(line_buffering=True)
dlog = logging.getLogger("dashboard")
dlog.setLevel(logging.INFO)
dlog.addHandler(_log_handler)
dlog.propagate = False  # Don't propagate to root logger — prevents
                        # Rich Live deadlock from concurrent console writes.
print(f"dashboard log: {_dashboard_log_path}", file=sys.stderr)

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POD_POLL_INTERVAL_S = 0.5
CD_POLL_INTERVAL_S = 0.5
REFRESH_HZ = 5

# ---------------------------------------------------------------------------
# Regexes for parsing result log lines
# ---------------------------------------------------------------------------

# Match: result(0@gb-nvl-156-compute14): ...
RESULT_RE = re.compile(r"result\((\d+)@([^)]+)\):\s*(.*)")
# Match: 1@node-g0-g1:818.3 GB/s (multi-GPU format)
# Groups: peer_idx, peer_node, remote_gpu, local_gpu, value
PEER_MULTI_RE = re.compile(
    r"(\d+)@(.+?)-g(\d+)-g(\d+):(\S+(?:\s+GB/s)?)"
)
# Match: 1@node:818.3 GB/s (single-GPU backward compat)
PEER_SINGLE_RE = re.compile(r"(\d+)@([^:]+):(\S+(?:\s+GB/s)?)")

# ---------------------------------------------------------------------------
# kubectl helpers
# ---------------------------------------------------------------------------

KUBECTL_TIMEOUT_S = 5


def kubectl_json(args, retries=2):
    """Run kubectl with -o json, return parsed JSON or None.

    Retries on timeout or error. Logs warnings on failure so kubectl
    hangs don't go unnoticed.
    """
    cmd_str = " ".join(["kubectl"] + args + ["-o", "json"])
    for attempt in range(1, retries + 1):
        try:
            out = subprocess.check_output(
                ["kubectl"] + args + ["-o", "json"],
                stderr=subprocess.DEVNULL, timeout=KUBECTL_TIMEOUT_S,
            )
            return json.loads(out)
        except subprocess.TimeoutExpired:
            dlog.warning("kubectl timeout (%ds) attempt %d/%d: %s",
                         KUBECTL_TIMEOUT_S, attempt, retries, cmd_str)
        except subprocess.CalledProcessError as exc:
            dlog.warning("kubectl error (rc=%d) attempt %d/%d: %s",
                         exc.returncode, attempt, retries, cmd_str)
        except json.JSONDecodeError:
            dlog.warning("kubectl invalid JSON attempt %d/%d: %s",
                         attempt, retries, cmd_str)
        if attempt < retries:
            time.sleep(0.5)
    return None


def get_node_ip(node_name):
    """Look up a node's internal IP via kubectl."""
    data = kubectl_json(["get", "node", node_name])
    if not data:
        return None
    for addr in data.get("status", {}).get("addresses", []):
        if addr.get("type") == "InternalIP":
            return addr["address"]
    return None


def get_atack_pods():
    """Return list of dicts: [{name, idx, node, status, ...}, ...]."""
    data = kubectl_json(["get", "pods", "-l", "app=atack"])
    if not data or "items" not in data:
        return []
    pods = []
    for item in data["items"]:
        name = item["metadata"]["name"]
        idx = name.rsplit("-", 1)[1]
        uid = item["metadata"].get("uid", "")
        node = item["spec"].get("nodeName", "?")
        created = item["metadata"].get("creationTimestamp", "")
        age = ""
        if created:
            try:
                ct = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
                delta = datetime.datetime.now(datetime.timezone.utc) - ct
                secs = int(delta.total_seconds())
                if secs < 60:
                    age = f"{secs}s"
                elif secs < 3600:
                    age = f"{secs // 60}m{secs % 60}s"
                else:
                    age = f"{secs // 3600}h{(secs % 3600) // 60}m"
            except Exception:
                age = "?"
        phase = item["status"].get("phase", "?")
        cstatuses = item["status"].get("containerStatuses", [])
        restart_count = 0
        if cstatuses:
            cs = cstatuses[0]
            restart_count = cs.get("restartCount", 0)
            if cs.get("ready"):
                status = "Ready"
            elif cs.get("state", {}).get("waiting"):
                status = cs["state"]["waiting"].get("reason", phase)
            else:
                status = phase
        else:
            status = phase

        # Kubelet's liveness probe status. Only look at current pod
        # conditions (not lastState, which is historical and persists
        # forever after a restart — causes false positives).
        liveness_failing = False
        if status == "Ready":
            liveness_failing = False  # Kubelet says Ready = liveness OK.
        else:
            # Pod not ready — check if it's specifically due to liveness.
            for cond in item["status"].get("conditions", []):
                if cond.get("type") == "Ready" and cond.get("status") == "False":
                    msg = cond.get("message", "")
                    if "liveness" in msg.lower():
                        liveness_failing = True

        pods.append({"name": name, "idx": idx, "uid": uid, "node": node, "age": age,
                      "status": status, "liveness_failing": liveness_failing,
                      "restart_count": restart_count,
                      "cuda_fatal": "", "direct_probe": "", "node_ip": None})
    return sorted(pods, key=lambda p: p["node"])


def probe_pod_healthz(pod):
    """Probe a pod's /healthz via its node IP + hostPort.

    Mutates the pod dict in place: sets direct_probe, cuda_fatal,
    liveness_failing.
    """
    node_ip = pod.get("node_ip")
    if not node_ip or pod["status"] != "Ready":
        return

    # Use a short socket timeout. urllib does not retry internally.
    # The timeout applies per socket operation (connect, read), not
    # wall-clock total, but with a small healthz response (~2 bytes)
    # the total is effectively bounded at ~0.5s.
    try:
        req = urllib.request.Request(f"http://{node_ip}:1337/healthz")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if body.strip() == "ok":
                pod["direct_probe"] = "ok"
            else:
                pod["direct_probe"] = body.strip()[:30]
                if "ILLEGAL_STATE" in body:
                    pod["cuda_fatal"] = body.strip()
                    pod["liveness_failing"] = True
    except urllib.error.HTTPError as exc:
        pod["direct_probe"] = f"HTTP {exc.code}"
        if exc.code == 500:
            pod["liveness_failing"] = True
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            if "ILLEGAL_STATE" in body:
                pod["cuda_fatal"] = body.strip()[:60]
    except urllib.error.URLError:
        pod["direct_probe"] = "unreachable"
    except OSError:
        pod["direct_probe"] = "unreachable"
    except Exception as exc:
        pod["direct_probe"] = str(exc)[:20]


def get_cd_daemons():
    """Return list of dicts for computedomain-daemon pods."""
    data = kubectl_json(["get", "pods", "-n", "nvidia-dra-driver-gpu"])
    if not data or "items" not in data:
        return []
    daemons = []
    for item in data["items"]:
        name = item["metadata"]["name"]
        if "computedomain-daemon" not in name:
            continue
        node = item["spec"].get("nodeName", "?")
        phase = item["status"].get("phase", "?")
        cstatuses = item["status"].get("containerStatuses", [])
        if cstatuses:
            cs = cstatuses[0]
            if cs.get("ready"):
                status = "Ready"
            elif cs.get("state", {}).get("waiting"):
                status = cs["state"]["waiting"].get("reason", phase)
            else:
                status = phase
        else:
            status = phase
        # Show only the unique tail, e.g. "…-35c58c59e163"
        parts = name.split("-")
        tail = parts[-1] if parts else name
        display_name = f"…-{tail}"
        daemons.append({"name": name, "display_name": display_name,
                         "node": node, "status": status})
    return sorted(daemons, key=lambda d: d["node"])


def get_cd_status():
    """Return dict with ComputeDomain status: {overall, nodes: [{index, name, status}]}."""
    data = kubectl_json(["get", "computedomain", "atack-compute-domain"])
    if not data or "status" not in data:
        return {"overall": "?", "nodes": []}
    st = data["status"]
    nodes = []
    for n in st.get("nodes", []):
        nodes.append({
            "index": n.get("index", "?"),
            "name": n.get("name", "?"),
            "status": n.get("status", "?"),
        })
    return {
        "overall": st.get("status", "?"),
        "nodes": sorted(nodes, key=lambda n: n["name"]),
    }


def get_statefulset_info():
    """Return StatefulSet metadata: gpus_per_node, generation, age."""
    data = kubectl_json(["get", "statefulset", "atack"])
    if not data:
        return None
    # Extract GPUS_PER_NODE from container env vars.
    gpus = 1
    try:
        envs = data["spec"]["template"]["spec"]["containers"][0].get("env", [])
        for e in envs:
            if e.get("name") == "GPUS_PER_NODE":
                gpus = int(e["value"])
                break
    except (KeyError, IndexError, ValueError):
        dlog.warning("could not extract GPUS_PER_NODE from StatefulSet env")
    created = data["metadata"].get("creationTimestamp", "")
    age = ""
    if created:
        try:
            ct = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
            delta = datetime.datetime.now(datetime.timezone.utc) - ct
            secs = int(delta.total_seconds())
            if secs < 60:
                age = f"{secs}s"
            elif secs < 3600:
                age = f"{secs // 60}m{secs % 60}s"
            else:
                age = f"{secs // 3600}h{(secs % 3600) // 60}m"
        except Exception:
            age = "?"
    return {
        "gpus_per_node": gpus,
        "generation": data["metadata"].get("generation", "?"),
        "age": age,
    }


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_statefulset(delta):
    """Scale atack statefulset by delta (+1 or -1). Runs in a background
    thread to avoid blocking the render loop."""
    def _do_scale():
        try:
            out = subprocess.check_output(
                ["kubectl", "get", "statefulset", "atack",
                 "-o", "jsonpath={.spec.replicas}"],
                stderr=subprocess.PIPE, timeout=5,
            )
            current = int(out.decode().strip())
            target = max(0, current + delta)
            subprocess.check_output(
                ["kubectl", "scale", "statefulset", "atack",
                 f"--replicas={target}"],
                stderr=subprocess.PIPE, timeout=5,
            )
        except Exception as exc:
            dlog.warning("scale error: %s", exc)

    threading.Thread(target=_do_scale, daemon=True).start()


# ---------------------------------------------------------------------------
# Log follower management
# ---------------------------------------------------------------------------

def start_log_follower(pod_name):
    proc = subprocess.Popen(
        ["kubectl", "logs", "-f", pod_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    fd = proc.stdout.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    return proc


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def shorten_node(name):
    """Default shortening: gb-nvl-156-compute15 -> compute15."""
    parts = name.split("-")
    return parts[-1] if len(parts) > 1 else name


def compact_node_names(names):
    """Strip the longest common prefix from a set of node names, returning
    a dict {original: shortened}. E.g. for gb-nvl-156-compute13,
    gb-nvl-156-compute14, gb-nvl-156-compute15 → {'...13': '13', ...}.
    Falls back to shorten_node() if only one name."""
    name_list = sorted(set(names))
    if len(name_list) <= 1:
        return {n: shorten_node(n) for n in names}

    # Find longest common prefix.
    prefix = name_list[0]
    for n in name_list[1:]:
        while not n.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                break

    # Strip prefix, but keep at least 2 characters.
    max_strip = max(0, min(len(n) for n in name_list) - 2)
    strip_len = min(len(prefix), max_strip)

    result = {}
    for n in names:
        short = n[strip_len:]
        if not short:
            short = n
        result[n] = short
    return result


def status_color(status):
    if status == "Ready":
        return "green"
    if status in ("ContainerCreating", "Pending", "PodInitializing"):
        return "yellow"
    if status in ("Terminated", "Succeeded"):
        return "dim"
    return "red"


def build_pods_table(atack_pods):
    table = Table(show_header=True, header_style="bold dim", box=None,
                  pad_edge=False, show_edge=False, padding=(0, 1))
    table.add_column("#", style="bold", width=3)
    table.add_column("Pod")
    table.add_column("Node")
    table.add_column("Age")
    table.add_column("Status")
    table.add_column("Restarts")
    table.add_column("Direct Probe")
    table.add_column("Liveness")
    table.add_column("Last Result")
    table.add_column("Fatal CUDA Error")

    for p in atack_pods:
        color = status_color(p["status"])
        probe = p.get("direct_probe", "")
        if probe == "ok":
            probe_text = Text("ok", style="green")
        elif probe:
            probe_text = Text(probe, style="red")
        else:
            probe_text = Text("")
        cuda_fatal = ""
        if p.get("cuda_fatal"):
            cuda_fatal = Text("ILLEGAL_STATE", style="red bold")
        last_result = p.get("last_result_ago")
        if last_result is not None:
            last_result_text = Text(f"{last_result}s ago")
        else:
            last_result_text = Text("")
        if p.get("liveness_failing"):
            liveness = Text("FAILING", style="red bold")
        elif p.get("status") == "Ready":
            liveness = Text("ok", style="green")
        else:
            liveness = Text("")
        restarts = p.get("restart_count", 0)
        restarts_text = Text(str(restarts), style="red" if restarts > 0 else "")
        table.add_row(p["idx"], p["name"], p["node"], p.get("age", ""),
                      Text(p["status"], style=color),
                      restarts_text, probe_text, liveness,
                      last_result_text, cuda_fatal)

    if not atack_pods:
        table.add_row("—", "no pods", "", "", "", "", "", "", "", "")

    return Panel(table, title="Workload Pods", title_align="left",
                 border_style="blue", padding=(0, 1))


def build_cd_table(cd_daemons):
    table = Table(show_header=True, header_style="bold dim", box=None,
                  pad_edge=False, show_edge=False, padding=(0, 1))
    table.add_column("Pod")
    table.add_column("Node")
    table.add_column("Status")

    for d in cd_daemons:
        color = status_color(d["status"])
        table.add_row(
            d["display_name"],
            d["node"],
            Text(d["status"], style=color),
        )

    if not cd_daemons:
        table.add_row("—", "none found", "")

    return Panel(table, title="ComputeDomain Daemon Pods (IMEX Daemons)", title_align="left",
                 border_style="blue", padding=(0, 1))


def build_cd_status_panel(cd_status):
    table = Table(show_header=True, header_style="bold dim", box=None,
                  pad_edge=False, show_edge=False, padding=(0, 1))
    table.add_column("#", width=3)
    table.add_column("Node")
    table.add_column("Status")
    table.add_column("Stale")

    for n in cd_status["nodes"]:
        is_stale = n["status"] == "stale"
        color = status_color(n["status"])
        stale_cell = Text("stale", style="red") if is_stale else Text("")
        status_text = Text(n["status"], style=color) if not is_stale else Text("—", style="dim")
        table.add_row(str(n["index"]), n["name"], status_text, stale_cell)

    if not cd_status["nodes"]:
        table.add_row("—", "none", "", "")

    return Panel(table, title="ComputeDomain Status", title_align="left",
                 border_style="blue", padding=(0, 1))


MATRIX_STALE_S = 15


def build_matrix_panel(latest_matrix, pod_nodes, live_matrix_keys,
                       matrix_timestamp, detected_poll_s, matrix_cell_times,
                       sts_info):
    title = "Bandwidth matrix (GB/s)"

    parts = []
    if matrix_timestamp:
        ago = int((datetime.datetime.now() - matrix_timestamp).total_seconds())
        parts.append(f"last update {ago}s ago")
    else:
        parts.append("no data yet")
    if detected_poll_s is not None:
        parts.append(f"benchmark repetition interval ~{detected_poll_s:.1f}s")
    if sts_info:
        parts.append(f"GPUs/node={sts_info['gpus_per_node']}")
        parts.append(f"StatefulSet age={sts_info['age']}")
    subtitle = "  |  ".join(parts)

    # Matrix keys are "pod_idx-gpu_idx", e.g. "0-0", "0-1", "1-0", "1-1".
    # Sort by pod index then GPU index.
    def sort_key(k):
        parts = k.split("-", 1)
        return (int(parts[0]), int(parts[1]))

    cols = sorted(live_matrix_keys, key=sort_key)
    if not cols:
        return Panel("(no pods)", title=title, title_align="left",
                     subtitle=subtitle, subtitle_align="left",
                     border_style="cyan")

    # Build column/row labels. For large matrices (>8 columns), use
    # compact node names that strip the common prefix to save space.
    all_nodes = [pod_nodes.get(c.split("-", 1)[0], "?") for c in cols]
    if len(cols) > 8:
        node_map = compact_node_names(all_nodes)
    else:
        node_map = {n: shorten_node(n) for n in all_nodes}

    col_headers = {}
    for c in cols:
        pod_idx, gpu_idx = c.split("-", 1)
        node = node_map.get(pod_nodes.get(pod_idx, "?"), "?")
        col_headers[c] = f"{pod_idx}-{node}-{gpu_idx}"

    # Adapt min column width to matrix size.
    min_col_width = 10 if len(cols) <= 8 else 7

    table = Table(show_header=True, header_style="bold", box=None,
                  pad_edge=False)
    table.add_column("", style="bold")
    for c in cols:
        table.add_column(col_headers[c], justify="right", min_width=min_col_width)

    for row_key in cols:
        pod_idx, gpu_idx = row_key.split("-", 1)
        node = node_map.get(pod_nodes.get(pod_idx, "?"), "?")
        row_label = f"{pod_idx}-{node}-{gpu_idx}"
        peers = latest_matrix.get(row_key, {})
        row_age = time.monotonic() - matrix_cell_times.get(row_key, 0)
        is_stale = row_age > MATRIX_STALE_S

        row_pod = row_key.split("-", 1)[0]
        cells = [row_label]
        for c in cols:
            col_pod = c.split("-", 1)[0]
            if col_pod == row_pod:
                cells.append(Text("—", style="dim"))
            elif not peers or is_stale:
                cells.append(Text("?", style="yellow"))
            else:
                val = peers.get(c, "?")
                if val == "?":
                    cells.append(Text("?", style="yellow"))
                elif any(s in val for s in ("err", "ERR", "MISMATCH",
                         "INVALID_HANDLE", "ILLEGAL_STATE", "lock-err")):
                    cells.append(Text(val, style="red bold"))
                else:
                    cells.append(Text(val, style="green"))
        table.add_row(*cells)

    return Panel(table, title=title, title_align="left",
                 subtitle=subtitle, subtitle_align="left",
                 border_style="cyan")


def build_header():
    return Text.assemble(
        (" U", "bold cyan"), " scale up  ",
        ("D", "bold cyan"), " scale down  ",
        ("Q", "bold cyan"), " quit",
    )


def build_layout(atack_pods, cd_daemons, cd_status, latest_matrix, pod_nodes,
                 live_matrix_keys, matrix_timestamp, detected_poll_s,
                 matrix_cell_times, sts_info):
    mid_row_h = max(len(cd_daemons), len(cd_status["nodes"]), 1) + 4

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=1),
        Layout(name="pods", size=len(atack_pods) + 4),
        Layout(name="mid", size=mid_row_h),
        Layout(name="matrix"),
    )
    layout["header"].update(build_header())
    layout["pods"].update(build_pods_table(atack_pods))
    layout["mid"].split_row(
        Layout(name="cd_daemons"),
        Layout(name="cd_status"),
    )
    layout["cd_daemons"].update(build_cd_table(cd_daemons))
    layout["cd_status"].update(build_cd_status_panel(cd_status))
    layout["matrix"].update(
        build_matrix_panel(latest_matrix, pod_nodes, live_matrix_keys,
                           matrix_timestamp, detected_poll_s,
                           matrix_cell_times, sts_info))
    return layout


# ---------------------------------------------------------------------------
# Per-panel data threads. Each panel has its own polling thread that does
# kubectl I/O independently. The render loop reads their latest results
# without blocking, so a slow API server never stalls the UI.
# ---------------------------------------------------------------------------

class PanelState:
    """Thread-safe container for one panel's data. Writers replace fields
    atomically (Python GIL). Readers see a consistent snapshot."""
    def __init__(self, **defaults):
        for k, v in defaults.items():
            setattr(self, k, v)
        self.last_update = None  # monotonic


def _linger_loop(fetch_fn, state, field, poll_s, linger_s, name_key="name"):
    """Generic poll-and-linger loop for a list of items.

    Fetches fresh data via ``fetch_fn()``, tracks vanished items for
    ``linger_s`` seconds with status='Terminated', writes the merged
    list to ``state.<field>``.
    """
    prev_names = set()
    gone = {}
    prev_items = []

    while True:
        now = time.monotonic()
        try:
            fresh = fetch_fn()
        except Exception:
            dlog.exception("_linger_loop(%s) failed:", field)
            time.sleep(poll_s)
            continue
        fresh_names = set(item[name_key] for item in fresh)

        for name in prev_names - fresh_names:
            if name not in gone:
                for item in prev_items:
                    if item[name_key] == name:
                        item["status"] = "Terminated"
                        gone[name] = {"data": item, "vanished_at": now}
                        break
        prev_names = fresh_names

        for name in list(gone.keys()):
            if now - gone[name]["vanished_at"] > linger_s:
                del gone[name]

        merged = fresh + [g["data"] for g in gone.values()]
        prev_items = merged
        setattr(state, field, merged)
        state.last_update = time.monotonic()
        time.sleep(poll_s)


def pods_poller(state, poll_s, linger_s):
    """Polls atack pods. Updates state.pods and state.live_pod_indices."""
    prev_names = set()
    gone = {}
    prev_pods = []
    node_ips = {}  # {node_name: ip} — resolved once per node, persists.
    last_sts_poll = 0

    while True:
        now = time.monotonic()

        # Poll StatefulSet info every ~5s.
        if now - last_sts_poll > 5:
            last_sts_poll = now
            try:
                state.sts_info = get_statefulset_info()
            except Exception as exc:
                dlog.warning("failed to poll StatefulSet info: %s", exc)
        try:
            fresh = get_atack_pods()
        except Exception:
            dlog.exception("pods_poller failed:")
            time.sleep(poll_s)
            continue

        # Resolve node IPs for any new nodes we haven't seen.
        for p in fresh:
            node = p["node"]
            if node not in node_ips and node != "?":
                ip = get_node_ip(node)
                if ip:
                    node_ips[node] = ip
                    dlog.warning("resolved node %s → %s", node, ip)
            p["node_ip"] = node_ips.get(p["node"])

        fresh_names = set(p["name"] for p in fresh)

        for name in prev_names - fresh_names:
            if name not in gone:
                for p in prev_pods:
                    if p["name"] == name:
                        p["status"] = "Terminated"
                        gone[name] = {"data": p, "vanished_at": now}
                        break
        prev_names = fresh_names

        for name in list(gone.keys()):
            if now - gone[name]["vanished_at"] > linger_s:
                del gone[name]

        # Probe healthz for each Ready pod via node IP.
        for p in fresh:
            probe_pod_healthz(p)

        merged = fresh + [g["data"] for g in gone.values()]
        prev_pods = merged
        state.pods = merged
        state.live_pod_indices = set(
            p["idx"] for p in fresh if p["status"] == "Ready"
        )
        state.last_update = time.monotonic()
        time.sleep(poll_s)


def cd_daemons_poller(state, poll_s, linger_s):
    """Polls ComputeDomain daemon pods."""
    _linger_loop(get_cd_daemons, state, "daemons", poll_s, linger_s)


def cd_status_poller(state, poll_s, linger_s):
    """Polls ComputeDomain status (node list)."""
    prev_node_names = set()
    gone = {}

    while True:
        now = time.monotonic()
        try:
            cd_status = get_cd_status()
        except Exception:
            dlog.exception("cd_status_poller failed:")
            time.sleep(poll_s)
            continue
        fresh_names = set(n["name"] for n in cd_status["nodes"])

        for nname in prev_node_names - fresh_names:
            if nname not in gone:
                gone[nname] = {
                    "data": {"index": "?", "name": nname, "status": "stale"},
                    "vanished_at": now,
                }
        prev_node_names = fresh_names

        for nname in list(gone.keys()):
            if now - gone[nname]["vanished_at"] > linger_s:
                del gone[nname]

        if gone:
            cd_status["nodes"] = cd_status["nodes"] + [
                g["data"] for g in gone.values()]

        state.status = cd_status
        state.last_update = time.monotonic()
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    pod_nodes = {}
    latest_matrix = {}       # {row_key: {col_key: value_str}} — updated directly
    matrix_cell_times = {}   # {row_key: monotonic time of last update}
    matrix_timestamp = None  # When we last got any result line
    detected_poll_s = None
    last_result_times = {}
    result_count = {}
    bootstrap_intervals = []
    live_matrix_keys = set()
    detected_gpus_per_node = 1

    followers = {}
    fd_to_pod = {}
    line_bufs = {}
    follower_backoff = {}  # {pod_name: monotonic time of last failure}
    last_follower_check = 0

    LINGER_S = 15.0

    # Each panel has its own state object and polling thread.
    pods_state = PanelState(pods=[], live_pod_indices=set(),
                            sts_info=None)
    cd_daemon_state = PanelState(daemons=[])
    cd_status_state = PanelState(status={"overall": "?", "nodes": []})

    threading.Thread(
        target=pods_poller,
        args=(pods_state, POD_POLL_INTERVAL_S, LINGER_S),
        daemon=True,
    ).start()
    threading.Thread(
        target=cd_daemons_poller,
        args=(cd_daemon_state, CD_POLL_INTERVAL_S, LINGER_S),
        daemon=True,
    ).start()
    threading.Thread(
        target=cd_status_poller,
        args=(cd_status_state, CD_POLL_INTERVAL_S, LINGER_S),
        daemon=True,
    ).start()

    console = Console()

    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    stdin_fd = sys.stdin.fileno()
    fcntl.fcntl(stdin_fd, fcntl.F_SETFL,
                fcntl.fcntl(stdin_fd, fcntl.F_GETFL) | os.O_NONBLOCK)

    last_heartbeat = 0

    with Live(console=console, refresh_per_second=REFRESH_HZ,
              screen=True) as live:
        try:
            while True:
                now = time.monotonic()

                # Heartbeat: log every 30s to confirm main loop is alive.
                if now - last_heartbeat > 30:
                    last_heartbeat = now
                    dlog.warning("heartbeat: main loop alive, %d followers, "
                                 "%d matrix keys",
                                 len(followers), len(latest_matrix))


                # --- Handle keyboard input ---
                try:
                    key = os.read(stdin_fd, 1)
                    if key in (b"q", b"Q", b"\x03"):
                        break
                    elif key in (b"u", b"U"):
                        scale_statefulset(+1)
                    elif key in (b"d", b"D"):
                        scale_statefulset(-1)
                except (BlockingIOError, OSError):
                    pass

                # --- Read shared state from panel threads (non-blocking) ---
                atack_pods = pods_state.pods
                live_pod_indices = pods_state.live_pod_indices
                sts_info = pods_state.sts_info
                gpus_per_node = sts_info["gpus_per_node"] if sts_info else detected_gpus_per_node
                live_matrix_keys = set(
                    f"{idx}-{g}" for idx in live_pod_indices
                    for g in range(gpus_per_node)
                )

                # --- Manage log followers ---
                # Only follow pods that are Ready (not Terminated/Pending).
                # Restart followers for pods whose subprocess has died.
                if now - last_follower_check > POD_POLL_INTERVAL_S:
                    last_follower_check = now
                    ready_names = set(
                        p["name"] for p in atack_pods if p["status"] == "Ready"
                    )

                    # Clean up dead followers. Track when they died to
                    # avoid spam-restarting (backoff: don't retry for 10s).
                    for pod_name in list(followers.keys()):
                        proc = followers[pod_name]
                        if proc.poll() is not None:
                            fd = proc.stdout.fileno()
                            fd_to_pod.pop(fd, None)
                            line_bufs.pop(fd, None)
                            del followers[pod_name]
                            follower_backoff[pod_name] = now
                            dlog.warning("log follower for %s died (rc=%s)",
                                         pod_name, proc.returncode)

                    # Remove followers for pods no longer Ready.
                    for pod_name in list(followers.keys()):
                        if pod_name not in ready_names:
                            proc = followers[pod_name]
                            proc.kill()
                            fd = proc.stdout.fileno()
                            fd_to_pod.pop(fd, None)
                            line_bufs.pop(fd, None)
                            del followers[pod_name]
                            dlog.warning("killed follower for non-ready pod %s",
                                         pod_name)

                    # Start followers for new Ready pods (with backoff).
                    for pod_name in ready_names - set(followers.keys()):
                        last_fail = follower_backoff.get(pod_name, 0)
                        if now - last_fail < 10:
                            continue
                        dlog.warning("starting log follower for %s", pod_name)
                        proc = start_log_follower(pod_name)
                        followers[pod_name] = proc
                        fd = proc.stdout.fileno()
                        fd_to_pod[fd] = pod_name
                        line_bufs[fd] = b""

                # --- Read log data ---
                stdout_fds = [p.stdout for p in followers.values()
                              if p.stdout]
                if stdout_fds:
                    try:
                        ready, _, _ = select.select(stdout_fds, [], [], 0)
                    except (ValueError, OSError) as exc:
                        # Bad fd from a dead process — clean up on next check.
                        dlog.warning("select failed: %s", exc)
                        ready = []
                    for fd_obj in ready:
                        fd = fd_obj.fileno()
                        try:
                            data = os.read(fd, 65536)
                        except (IOError, OSError):
                            continue
                        if not data:
                            continue

                        line_bufs[fd] = line_bufs.get(fd, b"") + data
                        while b"\n" in line_bufs[fd]:
                            raw_line, line_bufs[fd] = \
                                line_bufs[fd].split(b"\n", 1)
                            line_str = raw_line.decode("utf-8",
                                                       errors="replace")

                            m = RESULT_RE.search(line_str)
                            if not m:
                                continue

                            pod_idx = m.group(1)
                            node_name = m.group(2)
                            raw = m.group(3)
                            dlog.info("result line from pod %s", pod_idx)

                            pod_nodes[pod_idx] = node_name
                            matrix_timestamp = datetime.datetime.now()

                            # Parse peer entries and update latest_matrix
                            # directly. No round accumulation — every
                            # result line immediately updates the matrix.
                            multi_matches = list(PEER_MULTI_RE.finditer(raw))
                            if multi_matches:
                                seen_gpus = set()
                                for pm in multi_matches:
                                    peer_idx = pm.group(1)
                                    peer_node = pm.group(2)
                                    remote_gpu = pm.group(3)
                                    local_gpu = pm.group(4)
                                    val = pm.group(5)
                                    pod_nodes[peer_idx] = peer_node
                                    if val.endswith(" GB/s"):
                                        val = val[:-5]
                                    row_key = f"{pod_idx}-{local_gpu}"
                                    col_key = f"{peer_idx}-{remote_gpu}"
                                    if row_key not in latest_matrix:
                                        latest_matrix[row_key] = {}
                                    latest_matrix[row_key][col_key] = val
                                    matrix_cell_times[row_key] = time.monotonic()
                                    seen_gpus.add(int(local_gpu))
                                    seen_gpus.add(int(remote_gpu))
                                new_gpn = max(seen_gpus) + 1
                                if new_gpn > detected_gpus_per_node:
                                    detected_gpus_per_node = new_gpn
                                    live_matrix_keys = set(
                                        f"{idx}-{g}" for idx in live_pod_indices
                                        for g in range(detected_gpus_per_node)
                                    )
                            else:
                                # Single-GPU backward compat.
                                row_key = f"{pod_idx}-0"
                                if row_key not in latest_matrix:
                                    latest_matrix[row_key] = {}
                                for pm in PEER_SINGLE_RE.finditer(raw):
                                    peer_idx = pm.group(1)
                                    peer_node = pm.group(2)
                                    val = pm.group(3)
                                    pod_nodes[peer_idx] = peer_node
                                    if val.endswith(" GB/s"):
                                        val = val[:-5]
                                    latest_matrix[row_key][f"{peer_idx}-0"] = val
                                matrix_cell_times[row_key] = time.monotonic()

                            # Auto-detect poll interval from result cadence.
                            result_now = time.monotonic()
                            result_count[pod_idx] = result_count.get(pod_idx, 0) + 1
                            prev = last_result_times.get(pod_idx)
                            if prev and result_count[pod_idx] > 2:
                                interval = result_now - prev[0]
                                if 0.5 < interval < 60:
                                    if detected_poll_s is None:
                                        bootstrap_intervals.append(interval)
                                        if len(bootstrap_intervals) >= 4:
                                            s = sorted(bootstrap_intervals)
                                            detected_poll_s = s[len(s) // 2]
                                    else:
                                        detected_poll_s = 0.8 * detected_poll_s + 0.2 * interval
                            # Store (timestamp, uid) so we can detect pod replacements.
                            pod_uid = None
                            for p in pods_state.pods:
                                if p["idx"] == pod_idx:
                                    pod_uid = p.get("uid")
                                    break
                            last_result_times[pod_idx] = (result_now, pod_uid)

                # Remove matrix entries for pods no longer live.
                for key in list(latest_matrix.keys()):
                    if key not in live_matrix_keys:
                        del latest_matrix[key]
                        matrix_cell_times.pop(key, None)

                if not latest_matrix:
                    matrix_timestamp = None

                # --- Render ---
                # Enrich pod data with last-result age from log parsing.
                # Ignore stale entries where the pod UID changed (pod was
                # replaced — same index, different pod).
                now_mono = time.monotonic()
                display_pods = []
                for p in pods_state.pods:
                    p2 = dict(p)
                    entry = last_result_times.get(p["idx"])
                    if entry and entry[1] == p.get("uid"):
                        p2["last_result_ago"] = int(now_mono - entry[0])
                    else:
                        p2["last_result_ago"] = None
                    display_pods.append(p2)

                live.update(build_layout(
                    display_pods, cd_daemon_state.daemons,
                    cd_status_state.status, latest_matrix, pod_nodes,
                    live_matrix_keys, matrix_timestamp, detected_poll_s,
                    matrix_cell_times, sts_info))

                time.sleep(1.0 / REFRESH_HZ)

        except KeyboardInterrupt:
            dlog.warning("interrupted by user (Ctrl+C)")
        except Exception:
            dlog.exception("main loop crashed:")
        finally:
            # Always restore terminal and clean up, regardless of exit path.
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
            except Exception:
                dlog.warning("failed to restore terminal settings")
            for proc in followers.values():
                try:
                    proc.kill()
                except Exception:
                    dlog.warning("failed to kill follower process")

    # Print final status to stdout (visible after TUI exits).
    dlog.warning("dashboard exiting")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Last resort — if main() itself throws before the Live context.
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
