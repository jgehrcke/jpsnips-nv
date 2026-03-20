#!/usr/bin/env bash
set -euo pipefail

# Simulate a node replacement: cordon + drain a worker node running an
# atack pod, watch the StatefulSet reschedule it to a free node. This is
# a clean shutdown — the pod receives SIGTERM and performs graceful
# shutdown (evict broadcast, lock drain, CUDA cleanup).
#
# Requires: 3 running atack pods, 4 nodes (one free for reschedule).

POD_COUNT=$(kubectl get pods -l app=atack --no-headers | wc -l)
READY_COUNT=$(kubectl get pods -l app=atack --no-headers | grep -c "Running" || true)

if [[ "$POD_COUNT" -ne 3 || "$READY_COUNT" -ne 3 ]]; then
    echo "ERROR: expected 3 running atack pods, got ${READY_COUNT}/${POD_COUNT}"
    exit 1
fi
echo "3 pods running"
kubectl get pods -l app=atack -o wide

# Exclude control-plane nodes.
CP_NODES=$(kubectl get nodes -l node-role.kubernetes.io/control-plane \
    -o jsonpath='{.items[*].metadata.name}')
POD_NODES=$(kubectl get pods -l app=atack \
    -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}')

TARGET_NODE=""
for NODE in $POD_NODES; do
    IS_CP=false
    for CP in $CP_NODES; do
        [[ "$NODE" == "$CP" ]] && IS_CP=true && break
    done
    [[ "$IS_CP" == "false" ]] && TARGET_NODE="$NODE"
done

if [[ -z "$TARGET_NODE" ]]; then
    echo "ERROR: no suitable worker node found"
    exit 1
fi

TARGET_POD=$(kubectl get pods -l app=atack \
    -o jsonpath="{.items[?(@.spec.nodeName=='${TARGET_NODE}')].metadata.name}")

echo ""
echo "Identified node to drain: ${TARGET_NODE} (runs ${TARGET_POD})"
echo "Press Enter to drain, Ctrl+C to abort"
read -r

set -x
kubectl cordon "${TARGET_NODE}"
set +x

# Uncordon on any exit (success, error, Ctrl+C).
trap 'echo ""; echo "Uncordoning ${TARGET_NODE}"; kubectl uncordon "${TARGET_NODE}"' EXIT

set -x
kubectl drain "${TARGET_NODE}" --ignore-daemonsets --delete-emptydir-data --timeout=60s
set +x

echo ""
echo "Waiting for ${TARGET_POD} to reschedule..."
ATTEMPTS=0
while true; do
    ATTEMPTS=$((ATTEMPTS + 1))
    STATUS=$(kubectl get pod "${TARGET_POD}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
    NODE=$(kubectl get pod "${TARGET_POD}" -o jsonpath='{.spec.nodeName}' 2>/dev/null || echo "?")
    READY=$(kubectl get pod "${TARGET_POD}" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || echo "false")

    echo "  [${ATTEMPTS}] ${TARGET_POD}: ${STATUS} on ${NODE} ready=${READY}"

    if [[ "$READY" == "true" && "$NODE" != "$TARGET_NODE" ]]; then
        break
    fi
    if [[ "$ATTEMPTS" -ge 120 ]]; then
        echo "ERROR: timed out after 120 attempts"
        exit 1
    fi
    sleep 2
done

echo ""
echo "Node replacement complete: ${TARGET_POD} now on ${NODE}"
kubectl get pods -l app=atack -o wide
