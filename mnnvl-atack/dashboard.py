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

RESULT_RE = re.compile(r"result\((\d+)@([^)]+)\):\s*(.*)")
PEER_RE = re.compile(r"(\d+)@([^:]+):(\S+(?:\s+GB/s)?)")

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
    parts = name.split("-")
    return parts[-1] if len(parts) > 1 else name


def status_color(status):
    if status == "Ready":
        return "green"
    if status in ("ContainerCreating", "Pending", "PodInitializing"):
        return "yellow"
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

    for n in cd_status["nodes"]:
        color = status_color(n["status"])
        table.add_row(str(n["index"]), n["name"],
                      Text(n["status"], style=color))

    if not cd_status["nodes"]:
        table.add_row("—", "none", "")

    return Panel(table, title="ComputeDomain Status", title_align="left",
                 border_style="blue", padding=(0, 1))


def build_matrix_panel(latest_matrix, pod_nodes, live_pod_indices,
                       matrix_timestamp):
    title = "Bandwidth matrix (GB/s)"

    if matrix_timestamp:
        ago = int((datetime.datetime.now() - matrix_timestamp).total_seconds())
        subtitle = f"last update {ago}s ago"
    else:
        subtitle = "no data yet"

    cols = sorted(live_pod_indices, key=int)
    if not cols:
        return Panel("(no pods)", title=title, title_align="left",
                     subtitle=subtitle, subtitle_align="left",
                     border_style="cyan")

    # Build column headers.
    col_headers = {}
    for c in cols:
        node = shorten_node(pod_nodes.get(c, "?"))
        col_headers[c] = f"{c}-{node}"

    table = Table(show_header=True, header_style="bold", box=None,
                  pad_edge=False)
    table.add_column("", style="bold")  # Row label column.
    for c in cols:
        table.add_column(col_headers[c], justify="right", min_width=10)

    for pod_idx in cols:
        node = shorten_node(pod_nodes.get(pod_idx, "?"))
        row_label = f"{pod_idx}-{node}"
        peers = latest_matrix.get(pod_idx, {})

        cells = [row_label]
        for c in cols:
            if c == pod_idx:
                cells.append(Text("—", style="dim"))
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


def build_layout(atack_pods, cd_daemons, cd_status, latest_matrix, pod_nodes,
                 live_pod_indices, matrix_timestamp):
    mid_row_h = max(len(cd_daemons), len(cd_status["nodes"]), 1) + 4

    layout = Layout()
    layout.split_column(
        Layout(name="pods", size=len(atack_pods) + 4),
        Layout(name="mid", size=mid_row_h),
        Layout(name="matrix"),
    )
    layout["pods"].update(build_pods_table(atack_pods))
    layout["mid"].split_row(
        Layout(name="cd_daemons"),
        Layout(name="cd_status"),
    )
    layout["cd_daemons"].update(build_cd_table(cd_daemons))
    layout["cd_status"].update(build_cd_status_panel(cd_status))
    layout["matrix"].update(
        build_matrix_panel(latest_matrix, pod_nodes, live_pod_indices,
                           matrix_timestamp))
    return layout


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    pod_nodes = {}
    current_round = {}
    round_start = None
    latest_matrix = {}
    matrix_timestamp = None

    atack_pods = []
    cd_daemons = []
    cd_status = {"overall": "?", "nodes": []}
    live_pod_indices = set()

    followers = {}
    fd_to_pod = {}
    line_bufs = {}

    last_pod_poll = 0
    last_cd_poll = 0

    console = Console()

    with Live(console=console, refresh_per_second=REFRESH_HZ,
              screen=True) as live:
        try:
            while True:
                now = time.monotonic()

                # --- Poll atack pods ---
                if now - last_pod_poll > POD_POLL_INTERVAL_S:
                    last_pod_poll = now
                    atack_pods = get_atack_pods()
                    live_pod_indices = set(p["idx"] for p in atack_pods)

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
                    cd_daemons = get_cd_daemons()
                    cd_status = get_cd_status()

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

                            peers = {}
                            for pm in PEER_RE.finditer(raw):
                                peer_idx = pm.group(1)
                                peer_node = pm.group(2)
                                val = pm.group(3)
                                pod_nodes[peer_idx] = peer_node
                                if val.endswith(" GB/s"):
                                    val = val[:-5]
                                peers[peer_idx] = val

                            current_round[pod_idx] = peers
                            if round_start is None:
                                round_start = time.monotonic()

                            # Round complete.
                            if (live_pod_indices and
                                    live_pod_indices
                                    <= set(current_round.keys())):
                                latest_matrix = current_round
                                matrix_timestamp = datetime.datetime.now()
                                current_round = {}
                                round_start = None

                # Timeout fallback.
                if round_start is not None:
                    if time.monotonic() - round_start >= ROUND_WINDOW_S:
                        latest_matrix = current_round
                        matrix_timestamp = datetime.datetime.now()
                        current_round = {}
                        round_start = None

                # --- Render ---
                live.update(build_layout(
                    atack_pods, cd_daemons, cd_status, latest_matrix, pod_nodes,
                    live_pod_indices, matrix_timestamp))

                time.sleep(1.0 / REFRESH_HZ)

        except KeyboardInterrupt:
            pass
        finally:
            for proc in followers.values():
                proc.kill()


if __name__ == "__main__":
    main()
