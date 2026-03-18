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
import os
import re
import select
import subprocess
import sys
import termios
import time
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

def kubectl_json(args):
    """Run kubectl with -o json, return parsed JSON or None."""
    try:
        out = subprocess.check_output(
            ["kubectl"] + args + ["-o", "json"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return json.loads(out)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError):
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
        ip = item["status"].get("podIP", "?")
        pods.append({"name": name, "idx": idx, "node": node,
                      "status": status, "ip": ip})
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
    """Scale atack statefulset by delta (+1 or -1). Errors go to stderr."""
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
        print(f"scale error: {exc}", file=sys.stderr)


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
    table.add_column("Status")

    for p in atack_pods:
        color = status_color(p["status"])
        table.add_row(p["idx"], p["name"], p["node"],
                      Text(p["status"], style=color))

    if not atack_pods:
        table.add_row("—", "no pods", "", "")

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
# Main loop
# ---------------------------------------------------------------------------

def main():
    pod_nodes = {}
    current_round = {}
    round_start = None
    latest_matrix = {}       # {pod_idx: {peer_idx: value_str}}
    matrix_round_num = {}    # {pod_idx: round_number} — when each row was last updated
    round_counter = 0
    matrix_timestamp = None
    detected_poll_s = None      # Auto-detected poll interval from result cadence.
    last_result_times = {}      # {pod_idx: last_result_monotonic}
    result_count = {}           # {pod_idx: count} — skip first interval per pod
    bootstrap_intervals = []    # Collect initial intervals, seed EMA from median

    atack_pods = []
    cd_daemons = []
    cd_status = {"overall": "?", "nodes": []}
    live_pod_indices = set()
    live_matrix_keys = set()  # e.g. {"0-0", "0-1", "1-0", "1-1"}
    detected_gpus_per_node = 1

    followers = {}
    fd_to_pod = {}
    line_bufs = {}

    # Keep recently vanished pods visible for a few seconds.
    # {pod_key: {data, vanished_at}} where pod_key is (panel, name).
    LINGER_S = 15.0
    gone_pods = {}      # For atack pods: {name: {data, vanished_at}}
    gone_cd = {}        # For CD daemons: {name: {data, vanished_at}}
    gone_cd_nodes = {}  # For CD status nodes: {node_name: {data, vanished_at}}
    prev_atack_names = set()
    prev_cd_names = set()
    prev_cd_node_names = set()

    last_pod_poll = 0
    last_cd_poll = 0

    console = Console()

    # Put stdin in cbreak mode for single-keypress reading without
    # breaking terminal output (unlike raw mode which interferes with
    # rich's rendering).
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
                    if key in (b"q", b"Q", b"\x03"):  # q, Q, Ctrl-C
                        break
                    elif key in (b"u", b"U"):
                        scale_statefulset(+1)
                    elif key in (b"d", b"D"):
                        scale_statefulset(-1)
                except (BlockingIOError, OSError):
                    pass

                # --- Poll atack pods ---
                if now - last_pod_poll > POD_POLL_INTERVAL_S:
                    last_pod_poll = now
                    fresh_pods = get_atack_pods()
                    fresh_names = set(p["name"] for p in fresh_pods)

                    # Track newly vanished pods.
                    for name in prev_atack_names - fresh_names:
                        if name in gone_pods:
                            continue
                        # Find the last known data for this pod.
                        for p in atack_pods:
                            if p["name"] == name:
                                p["status"] = "Terminated"
                                gone_pods[name] = {"data": p, "vanished_at": now}
                                break
                    prev_atack_names = fresh_names

                    # Expire old lingering pods.
                    for name in list(gone_pods.keys()):
                        if now - gone_pods[name]["vanished_at"] > LINGER_S:
                            del gone_pods[name]

                    # Combine fresh + lingering.
                    atack_pods = fresh_pods + [g["data"] for g in gone_pods.values()]
                    live_pod_indices = set(
                        p["idx"] for p in fresh_pods if p["status"] == "Ready"
                    )
                    live_matrix_keys = set(
                        f"{idx}-{g}" for idx in live_pod_indices
                        for g in range(detected_gpus_per_node)
                    )

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

                # --- Poll CD daemons and status ---
                if now - last_cd_poll > CD_POLL_INTERVAL_S:
                    last_cd_poll = now
                    fresh_cd = get_cd_daemons()
                    fresh_cd_names = set(d["name"] for d in fresh_cd)

                    for name in prev_cd_names - fresh_cd_names:
                        if name in gone_cd:
                            continue
                        for d in cd_daemons:
                            if d["name"] == name:
                                d["status"] = "Terminated"
                                gone_cd[name] = {"data": d, "vanished_at": now}
                                break
                    prev_cd_names = fresh_cd_names

                    for name in list(gone_cd.keys()):
                        if now - gone_cd[name]["vanished_at"] > LINGER_S:
                            del gone_cd[name]

                    cd_daemons = fresh_cd + [g["data"] for g in gone_cd.values()]

                    cd_status = get_cd_status()
                    fresh_cd_node_names = set(n["name"] for n in cd_status["nodes"])

                    for nname in prev_cd_node_names - fresh_cd_node_names:
                        if nname in gone_cd_nodes:
                            continue
                        # Find last known data for this node.
                        for prev_nodes in [cd_status["nodes"]]:
                            pass  # Already gone from fresh data.
                        # Check previous cd_status stored nodes.
                        gone_cd_nodes[nname] = {
                            "data": {"index": "?", "name": nname, "status": "stale"},
                            "vanished_at": now,
                        }
                    prev_cd_node_names = fresh_cd_node_names

                    for nname in list(gone_cd_nodes.keys()):
                        if now - gone_cd_nodes[nname]["vanished_at"] > LINGER_S:
                            del gone_cd_nodes[nname]

                    # Merge lingering nodes into cd_status.
                    if gone_cd_nodes:
                        stale_nodes = [g["data"] for g in gone_cd_nodes.values()]
                        cd_status["nodes"] = cd_status["nodes"] + stale_nodes

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

                # --- Render ---
                live.update(build_layout(
                    atack_pods, cd_daemons, cd_status, latest_matrix, pod_nodes,
                    live_matrix_keys, matrix_timestamp, matrix_round_num,
                    round_counter, detected_poll_s))

                time.sleep(1.0 / REFRESH_HZ)

        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
            for proc in followers.values():
                proc.kill()


if __name__ == "__main__":
    main()
