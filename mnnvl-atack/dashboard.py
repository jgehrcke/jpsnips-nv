#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["rich"]
# ///
"""
TUI dashboard for atack — All-to-All CUDA Kubernetes test.

Displays four panels:
  - Pods: live status of atack StatefulSet pods
  - ComputeDomain daemons: status of computedomain-daemon pods
  - ComputeDomain status: node-level CD state
  - Bandwidth matrix: NVLink bandwidth (GB/s) between all GPU pairs

Each panel has its own polling thread. Bandwidth data is fetched from each
pod's /results HTTP endpoint via node IP + hostPort. Press Ctrl-C to quit.
"""

import concurrent.futures
import datetime
import fcntl
import json
import logging
import os
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


def _get_liveness_failure_events(pod_uids):
    """Return {pod_name: age_seconds} for recent liveness probe failures.

    Only returns events matching the current pod UIDs, so events from
    previous pod incarnations (same name, different UID) are ignored.
    """
    data = kubectl_json(["get", "events", "--field-selector",
                         "reason=Unhealthy"])
    if not data or "items" not in data:
        return {}
    result = {}
    now = datetime.datetime.now(datetime.timezone.utc)
    for ev in data["items"]:
        msg = ev.get("message", "")
        if "liveness" not in msg.lower():
            continue
        involved = ev.get("involvedObject", {})
        pod_name = involved.get("name", "")
        pod_uid = involved.get("uid", "")
        # Skip events from previous pod incarnations.
        if pod_uid and pod_uids.get(pod_name) != pod_uid:
            continue
        ts_str = ev.get("lastTimestamp") or ev.get("eventTime", "")
        if not ts_str or not pod_name:
            continue
        try:
            ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age_s = (now - ts).total_seconds()
            if pod_name not in result or age_s < result[pod_name]:
                result[pod_name] = age_s
        except Exception:
            continue
    return result


def get_atack_pods():
    """Return list of dicts: [{name, idx, node, status, ...}, ...]."""
    data = kubectl_json(["get", "pods", "-l", "app=atack"])
    if not data or "items" not in data:
        return []
    # Build UID map for filtering events to current pod incarnations.
    pod_uids = {item["metadata"]["name"]: item["metadata"].get("uid", "")
                for item in data["items"]}
    lp_events = _get_liveness_failure_events(pod_uids)
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

        # Age of most recent liveness probe failure event from kubelet.
        liveness_event_age = lp_events.get(name)  # seconds, or None.

        pods.append({"name": name, "idx": idx, "uid": uid, "node": node, "age": age,
                      "status": status, "lp_fail_age": liveness_event_age,
                      "restart_count": restart_count,
                      "cuda_fatal": "", "direct_probe": "", "node_ip": None})
    return sorted(pods, key=lambda p: p["node"])


def probe_pod_healthz(pod):
    """Probe a pod's /healthz via its node IP + hostPort.

    Mutates the pod dict in place: sets direct_probe, cuda_fatal,
    liveness_failing.
    """
    node_ip = pod.get("node_ip")
    if not node_ip:
        return

    # Use a short socket timeout. urllib does not retry internally.
    # Store probe time for freshness display.
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
    except urllib.error.HTTPError as exc:
        pod["direct_probe"] = f"HTTP {exc.code}"
        if exc.code == 500:
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
    pod["probe_time"] = time.monotonic()


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
        daemons.append({"name": name, "display_name": display_name,
                         "node": node, "status": status, "age": age})
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
    table.add_column("Age", min_width=6)
    table.add_column("Status", min_width=17)
    table.add_column("Restarts")
    table.add_column("Direct Liveness Probe")
    table.add_column("LP Fail Event")
    table.add_column("Last Result")
    for p in atack_pods:
        color = status_color(p["status"])
        probe = p.get("direct_probe", "")
        probe_age = ""
        pt = p.get("probe_time")
        if pt:
            probe_age = f" ({int(time.monotonic() - pt)}s ago)"
        if probe == "ok":
            probe_text = Text(f"ok{probe_age}", style="green")
        elif probe:
            probe_text = Text(f"{probe}{probe_age}", style="red")
        else:
            probe_text = Text("")
        last_result = p.get("last_result_ago")
        if last_result is not None:
            last_result_text = Text(f"{last_result}s ago")
        else:
            last_result_text = Text("")
        lp_age = p.get("lp_fail_age")
        if lp_age is not None and lp_age < 30:
            liveness = Text(f"{int(lp_age)}s ago", style="red")
        elif lp_age is not None:
            liveness = Text(f"{int(lp_age)}s ago", style="dim")
        else:
            liveness = Text("")
        restarts = p.get("restart_count", 0)
        restarts_text = Text(str(restarts), style="red" if restarts > 0 else "")
        table.add_row(p["idx"], p["name"], p["node"], p.get("age", ""),
                      Text(p["status"], style=color),
                      restarts_text, probe_text, liveness,
                      last_result_text)

    if not atack_pods:
        table.add_row("—", "no pods", "", "", "", "", "", "", "")

    return Panel(table, title="Workload Pods", title_align="left",
                 border_style="blue", padding=(0, 1))


def build_cd_table(cd_daemons):
    table = Table(show_header=True, header_style="bold dim", box=None,
                  pad_edge=False, show_edge=False, padding=(0, 1))
    table.add_column("Pod")
    table.add_column("Node")
    table.add_column("Age")
    table.add_column("Status")

    for d in cd_daemons:
        color = status_color(d["status"])
        table.add_row(
            d["display_name"],
            d["node"],
            d.get("age", ""),
            Text(d["status"], style=color),
        )

    if not cd_daemons:
        table.add_row("—", "none found", "", "")

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
            elif not peers:
                cells.append(Text("?", style="yellow"))
            else:
                val = peers.get(c, "?")
                if val == "?":
                    cells.append(Text("?", style="yellow"))
                elif any(s in val for s in ("err", "ERR", "MISMATCH",
                         "INVALID_HANDLE", "ILLEGAL_STATE",
                         "LAUNCH_FAILED", "lock-err",
                         "unreachable")):
                    cells.append(Text(val, style="red bold"))
                elif is_stale:
                    cells.append(Text(val, style="dim"))
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

        # Probe healthz for all pods with a node IP — not just Ready ones.
        # A failing pod still has its HTTP server running until kubelet
        # kills the container, so we can show the probe result during the
        # liveness failure window.
        probeable = [p for p in fresh if p.get("node_ip")]
        if probeable:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(probeable)) as pool:
                pool.map(probe_pod_healthz, probeable)

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


def _fetch_pod_results(node_ip):
    """Fetch /results from a pod via node IP. Returns parsed JSON or None."""
    try:
        req = urllib.request.Request(f"http://{node_ip}:1337/results")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _pod_results_loop(state, idx, node_ip, uid, poll_s, stop_event):
    """Dedicated polling loop for one pod. Exits when stop_event is set."""
    fetch_count = 0
    prev_timestamp = None       # Result timestamp from previous fetch.
    prev_timestamp_mono = None  # Monotonic time when that timestamp first appeared.
    while not stop_event.is_set():
        now = time.monotonic()
        data = _fetch_pod_results(node_ip)
        fetch_count += 1
        if fetch_count <= 3 or fetch_count % 50 == 0:
            dlog.warning("pod %s fetch #%d: data=%s", idx, fetch_count,
                         "yes" if data else "no")
        results_list = data.get("results", []) if data else []
        if results_list:
            # Most recent entry (may be in-progress).
            latest = results_list[-1]
            age_s = latest.get("age_s", 999)
            latest_ts = latest.get("timestamp", "")

            # Always update age (freshness indicator).
            state.last_result_times[idx] = (age_s, uid)

            # Only update the matrix when the latest timestamp changed.
            if latest_ts and latest_ts != prev_timestamp:
                # Merge all results: iterate oldest to newest so the
                # most recent value for each cell wins. This way, if the
                # latest round is missing a cell (peer unreachable), the
                # error or value from a previous round shows through.
                for result in results_list:
                    for b in result.get("benchmarks", []):
                        peer_idx = str(b["peer_idx"])
                        peer_node = b["peer_node"]
                        remote_gpu = str(b["remote_gpu"])
                        local_gpu = str(b["local_gpu"])
                        val = b["value"]
                        if val.endswith(" GB/s"):
                            val = val[:-5]
                        row_key = f"{idx}-{local_gpu}"
                        col_key = f"{peer_idx}-{remote_gpu}"
                        if row_key not in state.matrix:
                            state.matrix[row_key] = {}
                        state.matrix[row_key][col_key] = val
                        state.cell_times[row_key] = now
                        state.pod_nodes[idx] = data.get("node_name", "?")
                        state.pod_nodes[peer_idx] = peer_node

                state.timestamp = datetime.datetime.now()
                state.last_update = now

                # Detect benchmark repetition interval from the time
                # between result changes.
                if prev_timestamp_mono is not None:
                    interval = now - prev_timestamp_mono
                    if 0.5 < interval < 120:
                        if state.detected_poll_s is None:
                            state.detected_poll_s = interval
                        else:
                            state.detected_poll_s = (
                                0.8 * state.detected_poll_s + 0.2 * interval)
                prev_timestamp = latest_ts
                prev_timestamp_mono = now

        stop_event.wait(poll_s)
    dlog.warning("results poller for pod %s (uid=%s) exiting", idx, uid)


def results_poller_spawner(state, pods_state, poll_s):
    """Spawns one dedicated polling thread per pod (identified by UID).

    Watches pods_state for new pods with node IPs and starts a
    _pod_results_loop thread for each. When a pod disappears from
    pods_state, its stop_event is set so the thread exits cleanly.
    """
    active = {}  # {(idx, uid): (Thread, stop_event)}
    while True:
        pods = pods_state.pods
        live_keys = set()
        for p in pods:
            node_ip = p.get("node_ip")
            if not node_ip:
                continue
            key = (p["idx"], p.get("uid"))
            live_keys.add(key)
            if key in active:
                continue
            stop = threading.Event()
            t = threading.Thread(
                target=_pod_results_loop,
                args=(state, p["idx"], node_ip, p.get("uid"), poll_s, stop),
                daemon=True,
            )
            t.start()
            active[key] = (t, stop)
            dlog.warning("started results poller for pod %s (uid=%s)",
                         p["idx"], p.get("uid"))

        # Stop threads for pods that are gone.
        for key in list(active):
            if key not in live_keys:
                _, stop = active.pop(key)
                stop.set()
                dlog.warning("stopping results poller for pod %s (uid=%s)",
                             key[0], key[1])

        time.sleep(1.0)


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
    live_matrix_keys = set()

    LINGER_S = 15.0
    RESULTS_POLL_S = 0.5

    # Each panel has its own state object and polling thread.
    pods_state = PanelState(pods=[], live_pod_indices=set(),
                            sts_info=None)
    cd_daemon_state = PanelState(daemons=[])
    cd_status_state = PanelState(status={"overall": "?", "nodes": []})
    results_state = PanelState(
        matrix={},            # {row_key: {col_key: value_str}}
        cell_times={},        # {row_key: monotonic time}
        timestamp=None,       # datetime of last result
        detected_poll_s=None,
        pod_nodes={},         # {pod_idx: node_name}
        last_result_times={}, # {pod_idx: (monotonic, uid)}
    )

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
    threading.Thread(
        target=results_poller_spawner,
        args=(results_state, pods_state, RESULTS_POLL_S),
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
                    dlog.warning("heartbeat: main loop alive, "
                                 "%d matrix keys",
                                 len(results_state.matrix))


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
                gpus_per_node = sts_info["gpus_per_node"] if sts_info else 1
                live_matrix_keys = set(
                    f"{idx}-{g}" for idx in live_pod_indices
                    for g in range(gpus_per_node)
                )

                # Read matrix data from results poller.
                latest_matrix = results_state.matrix
                matrix_cell_times = results_state.cell_times
                matrix_timestamp = results_state.timestamp
                detected_poll_s = results_state.detected_poll_s
                pod_nodes = results_state.pod_nodes

                # Remove matrix entries for pods no longer live.
                for key in list(latest_matrix.keys()):
                    if key not in live_matrix_keys:
                        del latest_matrix[key]
                        matrix_cell_times.pop(key, None)

                if not latest_matrix:
                    matrix_timestamp = None

                # --- Render ---
                now_mono = time.monotonic()
                display_pods = []
                for p in atack_pods:
                    p2 = dict(p)
                    entry = results_state.last_result_times.get(p["idx"])
                    if entry and entry[1] == p.get("uid"):
                        p2["last_result_ago"] = int(entry[0])
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

    # Print final status to stdout (visible after TUI exits).
    dlog.warning("dashboard exiting")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Last resort — if main() itself throws before the Live context.
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
