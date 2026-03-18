#!/usr/bin/env python3
"""
Follow atack result lines and display a matrix view.

Manages kubectl logs -f subprocesses per pod, automatically starting new
log followers when pods appear. Displays a bandwidth matrix (GB/s).

Uses a simple time-window approach: after the first result line arrives,
waits ROUND_WINDOW_S for more lines, then prints whatever was collected.
This avoids brittle round-completion logic that breaks during scale-up/down
when pods appear or disappear mid-round.

Example output:
    ------------------------------------------------------
                     0-compute13  1-compute16  2-compute15
    0-compute13                —        818.9        817.2
    1-compute16            818.0            —        816.5
    2-compute15            817.7        815.8            —
"""

import fcntl
import os
import re
import select
import subprocess
import sys
import time

# Match: result(1@gb-nvl-156-compute14): 0@gb-nvl-156-compute15:818.0 GB/s
RESULT_RE = re.compile(r"result\((\d+)@([^)]+)\):\s*(.*)")
# Match: 1@gb-nvl-156-compute14:818.0 GB/s  or  1@node:err
PEER_RE = re.compile(r"(\d+)@([^:]+):(\S+(?:\s+GB/s)?)")

# Time window: after the first result line in a round, wait this long
# for more lines before printing. Should be long enough to collect all
# pods' results (they poll at roughly the same interval) but short enough
# to feel responsive.
ROUND_WINDOW_S = 2.5
POD_POLL_INTERVAL_S = 3.0


def get_running_pods():
    """Return set of pod names like {'atack-0', 'atack-1'}."""
    try:
        out = subprocess.check_output(
            ["kubectl", "get", "pods", "-l", "app=atack",
             "-o", "jsonpath={.items[*].metadata.name}"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        names = out.decode().split()
        return set(names)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return set()


def start_log_follower(pod_name):
    """Start kubectl logs -f for a single pod, return the Popen object."""
    proc = subprocess.Popen(
        ["kubectl", "logs", "-f", pod_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    fd = proc.stdout.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    return proc


def shorten_node(name):
    """Shorten node name, e.g. gb-nvl-156-compute15 -> compute15."""
    parts = name.split("-")
    if len(parts) > 1:
        return parts[-1]
    return name


def print_round(results, pod_nodes, live_pod_indices):
    """Print the matrix. Rows and columns are determined by live_pod_indices
    (derived from kubectl get pods), not from the round data."""
    cols = sorted(live_pod_indices, key=int)
    if not cols:
        return

    col_headers = {}
    for c in cols:
        node = shorten_node(pod_nodes.get(c, "?"))
        col_headers[c] = f"{c}-{node}"
    col_width = max((len(h) for h in col_headers.values()), default=8) + 2

    row_labels = {}
    for pod_idx in cols:
        node = shorten_node(pod_nodes.get(pod_idx, "?"))
        row_labels[pod_idx] = f"  {pod_idx}-{node}"
    label_width = max((len(l) for l in row_labels.values()), default=10) + 2

    separator = "-" * (label_width + col_width * len(cols))
    print(separator)

    header = " " * label_width
    for c in cols:
        header += f"{col_headers[c]:>{col_width}}"
    print(header)

    for pod_idx in cols:
        label = row_labels[pod_idx]
        row = f"{label:{label_width}}"
        peers = results.get(pod_idx, {})
        for c in cols:
            if c == pod_idx:
                cell = "\u2014"
            else:
                cell = peers.get(c, "?")
            row += f"{cell:>{col_width}}"
        print(row)

    sys.stdout.flush()


def main():
    pod_nodes = {}           # {pod_idx: node_name}, kept across rounds
    current_round = {}       # {pod_idx: {peer_idx: value_str}}
    round_start = None
    ever_had_pods = False
    live_pod_indices = set() # Pod indices from kubectl get pods

    followers = {}     # {pod_name: Popen}
    fd_to_pod = {}     # {fileno: pod_name}
    line_bufs = {}     # {fileno: bytes}

    last_pod_check = 0

    try:
        while True:
            now = time.monotonic()

            # Periodically discover new pods and start log followers.
            if now - last_pod_check > POD_POLL_INTERVAL_S:
                last_pod_check = now
                current_pods = get_running_pods()
                live_pod_indices = set(
                    name.rsplit("-", 1)[1] for name in current_pods
                )
                for pod_name in current_pods - set(followers.keys()):
                    proc = start_log_follower(pod_name)
                    followers[pod_name] = proc
                    fd = proc.stdout.fileno()
                    fd_to_pod[fd] = pod_name
                    line_bufs[fd] = b""
                # Clean up dead followers.
                for pod_name in list(followers.keys()):
                    proc = followers[pod_name]
                    if proc.poll() is not None:
                        fd = proc.stdout.fileno()
                        fd_to_pod.pop(fd, None)
                        line_bufs.pop(fd, None)
                        del followers[pod_name]

            if not followers and ever_had_pods:
                print("all pods terminated")
                sys.stdout.flush()
                return
            if not followers:
                time.sleep(1)
                continue
            ever_had_pods = True

            # select() with appropriate timeout.
            timeout = POD_POLL_INTERVAL_S
            if round_start is not None:
                remaining = ROUND_WINDOW_S - (now - round_start)
                timeout = max(0, min(timeout, remaining))

            stdout_fds = [p.stdout for p in followers.values() if p.stdout]
            if not stdout_fds:
                time.sleep(0.5)
                continue

            ready, _, _ = select.select(stdout_fds, [], [], timeout)

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
                    line, line_bufs[fd] = line_bufs[fd].split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="replace")

                    m = RESULT_RE.search(line_str)
                    if not m:
                        continue

                    pod_idx = m.group(1)
                    node_name = m.group(2)
                    raw = m.group(3)

                    prev_node = pod_nodes.get(pod_idx)
                    pod_nodes[pod_idx] = node_name
                    if prev_node is not None and prev_node != node_name:
                        print(f"  *** pod {pod_idx} moved: {prev_node} -> {node_name} ***")
                        sys.stdout.flush()

                    peers = {}
                    for pm in PEER_RE.finditer(raw):
                        peer_idx = pm.group(1)
                        peer_node = pm.group(2)
                        val = pm.group(3)
                        pod_nodes[peer_idx] = peer_node
                        if val.endswith(" GB/s"):
                            val = val[:-5]
                        peers[peer_idx] = val

                    # Keep the latest result per pod within the window.
                    current_round[pod_idx] = peers

                    if round_start is None:
                        round_start = time.monotonic()

                    # Round complete: heard from all live pods.
                    if (live_pod_indices
                            and live_pod_indices <= set(current_round.keys())):
                        print_round(current_round, pod_nodes, live_pod_indices)
                        current_round = {}
                        round_start = None

            # Timeout fallback: print whatever we have.
            if round_start is not None:
                if time.monotonic() - round_start >= ROUND_WINDOW_S:
                    print_round(current_round, pod_nodes, live_pod_indices)
                    current_round = {}
                    round_start = None

    except KeyboardInterrupt:
        pass
    finally:
        for proc in followers.values():
            proc.kill()


if __name__ == "__main__":
    main()
