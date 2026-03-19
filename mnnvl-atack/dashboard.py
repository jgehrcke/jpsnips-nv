#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["rich", "requests"]
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

import requests as requests_lib

# Dashboard diagnostics go to stderr so they survive TUI crashes and
# don't interfere with Rich's screen rendering on stdout.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="%(asctime)s dashboard %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
dlog = logging.getLogger("dashboard")
import tty

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROUND_WINDOW_S = 2.5
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


def get_atack_pods():
    """Return list of dicts: [{name, idx, node, status, ip}, ...]."""
    data = kubectl_json(["get", "pods", "-l", "app=atack"])
    if not data or "items" not in data:
        return []
    pods = []
    for item in data["items"]:
        name = item["metadata"]["name"]
        idx = name.rsplit("-", 1)[1]
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
        liveness_failing = False
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

        # Detect liveness probe failure from pod conditions.
        for cond in item["status"].get("conditions", []):
            if cond.get("type") == "ContainersReady" and cond.get("status") == "False":
                reason = cond.get("reason", "")
                if reason:
                    liveness_failing = True

        # Also check if container was recently killed by liveness probe.
        if cstatuses:
            last_state = cstatuses[0].get("lastState", {})
            terminated = last_state.get("terminated", {})
            if terminated.get("reason") == "OOMKilled" or (
                    terminated.get("exitCode", 0) != 0 and restart_count > 0):
                liveness_failing = True

        # Probe /healthz to detect CUDA_ERROR_ILLEGAL_STATE.
        cuda_fatal = ""
        pod_ip = item["status"].get("podIP")
        if pod_ip and status == "Ready":
            try:
                resp = requests_lib.get(f"http://{pod_ip}:1337/healthz", timeout=(0.5, 1))
                if resp.status_code == 500 and "ILLEGAL_STATE" in resp.text:
                    cuda_fatal = resp.text
                    liveness_failing = True
            except Exception:
                pass

        pods.append({"name": name, "idx": idx, "node": node, "age": age,
                      "status": status, "liveness_failing": liveness_failing,
                      "restart_count": restart_count, "cuda_fatal": cuda_fatal})
    return sorted(pods, key=lambda p: p["node"])


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
    table.add_column("Liveness")
    table.add_column("Fatal CUDA Error")

    for p in atack_pods:
        color = status_color(p["status"])
        if p.get("liveness_failing"):
            liveness = Text("FAILING", style="red bold")
        elif p.get("status") == "Ready":
            liveness = Text("ok", style="green")
        else:
            liveness = ""
        cuda_fatal = ""
        if p.get("cuda_fatal"):
            cuda_fatal = Text("ILLEGAL_STATE", style="red bold")
        table.add_row(p["idx"], p["name"], p["node"], p.get("age", ""),
                      Text(p["status"], style=color),
                      liveness, cuda_fatal)

    if not atack_pods:
        table.add_row("—", "no pods", "", "", "", "", "")

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


def build_matrix_panel(latest_matrix, pod_nodes, live_matrix_keys,
                       matrix_timestamp, matrix_round_num, round_counter,
                       detected_poll_s):
    title = "Bandwidth matrix (GB/s)"

    parts = []
    if matrix_timestamp:
        ago = int((datetime.datetime.now() - matrix_timestamp).total_seconds())
        parts.append(f"last update {ago}s ago")
    else:
        parts.append("no data yet")
    if detected_poll_s is not None:
        parts.append(f"benchmark repetition interval ~{detected_poll_s:.1f}s")
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
        row_round = matrix_round_num.get(row_key, 0)
        is_stale = round_counter - row_round > 1

        row_pod = row_key.split("-", 1)[0]
        cells = [row_label]
        for c in cols:
            col_pod = c.split("-", 1)[0]
            # Same pod = intra-node pair, no cross-pod measurement.
            if col_pod == row_pod:
                cells.append(Text("—", style="dim"))
            elif not peers:
                cells.append(Text("?", style="yellow"))
            elif is_stale:
                val = peers.get(c, "?")
                cells.append(Text(f"{val} !", style="dim"))
            else:
                val = peers.get(c, "?")
                if val == "?":
                    cells.append(Text("?", style="yellow"))
                elif "err" in val.lower() or "MISMATCH" in val:
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
                 live_matrix_keys, matrix_timestamp, matrix_round_num,
                 round_counter, detected_poll_s):
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
                           matrix_timestamp, matrix_round_num, round_counter,
                           detected_poll_s))
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

    while True:
        now = time.monotonic()
        try:
            fresh = get_atack_pods()
        except Exception:
            dlog.exception("pods_poller failed:")
            time.sleep(poll_s)
            continue
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
    current_round = {}
    round_start = None
    latest_matrix = {}
    matrix_round_num = {}
    round_counter = 0
    matrix_timestamp = None
    detected_poll_s = None
    last_result_times = {}
    result_count = {}
    bootstrap_intervals = []
    live_matrix_keys = set()
    detected_gpus_per_node = 1

    followers = {}
    fd_to_pod = {}
    line_bufs = {}
    last_follower_check = 0

    LINGER_S = 15.0

    # Each panel has its own state object and polling thread.
    pods_state = PanelState(pods=[], live_pod_indices=set())
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

    with Live(console=console, refresh_per_second=REFRESH_HZ,
              screen=True) as live:
        try:
            while True:
                now = time.monotonic()

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
                live_matrix_keys = set(
                    f"{idx}-{g}" for idx in live_pod_indices
                    for g in range(detected_gpus_per_node)
                )

                # --- Manage log followers ---
                if now - last_follower_check > POD_POLL_INTERVAL_S:
                    last_follower_check = now
                    current_names = set(p["name"] for p in atack_pods)
                    for pod_name in current_names - set(followers.keys()):
                        proc = start_log_follower(pod_name)
                        followers[pod_name] = proc
                        fd = proc.stdout.fileno()
                        fd_to_pod[fd] = pod_name
                        line_bufs[fd] = b""
                    for pod_name in list(followers.keys()):
                        proc = followers[pod_name]
                        if proc.poll() is not None:
                            fd = proc.stdout.fileno()
                            fd_to_pod.pop(fd, None)
                            line_bufs.pop(fd, None)
                            del followers[pod_name]

                # --- Read log data ---
                stdout_fds = [p.stdout for p in followers.values()
                              if p.stdout]
                if stdout_fds:
                    ready, _, _ = select.select(stdout_fds, [], [], 0)
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

                            pod_nodes[pod_idx] = node_name

                            # Parse peer entries. Multi-GPU format:
                            #   1@node-g0-g1:818.3 GB/s
                            #   → col_key="1-0" (peer+remote_gpu)
                            #   → row_key="reporter-local_gpu"
                            # Single-GPU fallback:
                            #   1@node:818.3 GB/s
                            #   → col_key="1-0", row_key="reporter-0"
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
                                    if row_key not in current_round:
                                        current_round[row_key] = {}
                                    current_round[row_key][col_key] = val
                                    seen_gpus.add(int(local_gpu))
                                    seen_gpus.add(int(remote_gpu))
                                new_gpn = max(seen_gpus) + 1
                                if new_gpn > detected_gpus_per_node:
                                    detected_gpus_per_node = new_gpn
                                    # Rebuild matrix keys with new GPU count.
                                    live_matrix_keys = set(
                                        f"{idx}-{g}" for idx in live_pod_indices
                                        for g in range(detected_gpus_per_node)
                                    )
                            else:
                                # Single-GPU backward compat.
                                peers = {}
                                for pm in PEER_SINGLE_RE.finditer(raw):
                                    peer_idx = pm.group(1)
                                    peer_node = pm.group(2)
                                    val = pm.group(3)
                                    pod_nodes[peer_idx] = peer_node
                                    if val.endswith(" GB/s"):
                                        val = val[:-5]
                                    peers[f"{peer_idx}-0"] = val
                                current_round[f"{pod_idx}-0"] = peers
                            if round_start is None:
                                round_start = time.monotonic()

                            # Auto-detect poll interval from result cadence.
                            # Skip first two results per pod — the initial
                            # intervals are unreliable due to log follower
                            # buffering and startup timing.
                            result_now = time.monotonic()
                            result_count[pod_idx] = result_count.get(pod_idx, 0) + 1
                            if pod_idx in last_result_times and result_count[pod_idx] > 2:
                                interval = result_now - last_result_times[pod_idx]
                                if 0.5 < interval < 60:
                                    if detected_poll_s is None:
                                        # Collect intervals, seed from median
                                        # once we have enough samples.
                                        bootstrap_intervals.append(interval)
                                        if len(bootstrap_intervals) >= 4:
                                            s = sorted(bootstrap_intervals)
                                            detected_poll_s = s[len(s) // 2]
                                    else:
                                        detected_poll_s = 0.8 * detected_poll_s + 0.2 * interval
                            last_result_times[pod_idx] = result_now

                            # Round complete when all matrix keys reported.
                            if (live_matrix_keys and
                                    live_matrix_keys
                                    <= set(current_round.keys())):
                                round_counter += 1
                                latest_matrix.update(current_round)
                                for pod_idx in current_round:
                                    matrix_round_num[pod_idx] = round_counter
                                matrix_timestamp = datetime.datetime.now()
                                current_round = {}
                                round_start = None

                # Timeout fallback: merge what we have so far.
                # Use 3x detected poll interval, or ROUND_WINDOW_S as default.
                round_timeout = (detected_poll_s * 3) if detected_poll_s else ROUND_WINDOW_S
                if round_start is not None:
                    if time.monotonic() - round_start >= round_timeout:
                        round_counter += 1
                        latest_matrix.update(current_round)
                        for pod_idx in current_round:
                            matrix_round_num[pod_idx] = round_counter
                        matrix_timestamp = datetime.datetime.now()
                        current_round = {}
                        round_start = None

                # Remove matrix entries that are no longer live.
                for key in list(latest_matrix.keys()):
                    if key not in live_matrix_keys:
                        del latest_matrix[key]
                        matrix_round_num.pop(key, None)

                # Reset timestamp when matrix is empty (all pods gone).
                if not latest_matrix:
                    matrix_timestamp = None

                # --- Render ---
                live.update(build_layout(
                    pods_state.pods, cd_daemon_state.daemons,
                    cd_status_state.status, latest_matrix, pod_nodes,
                    live_matrix_keys, matrix_timestamp, matrix_round_num,
                    round_counter, detected_poll_s))

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
                pass
            for proc in followers.values():
                try:
                    proc.kill()
                except Exception:
                    pass

    # Print final status to stdout (visible after TUI exits).
    dlog.warning("dashboard exiting")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Last resort — if main() itself throws before the Live context.
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
