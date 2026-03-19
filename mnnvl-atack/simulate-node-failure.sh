#!/usr/bin/env bash
set -euo pipefail

# Simulate a sudden node failure by force-killing an atack pod (no SIGTERM,
# no graceful shutdown). The StatefulSet recreates the pod, which should
# land on a different node via anti-affinity. Tests IMEX daemon recovery
# and peer resilience to abrupt handle invalidation.
#
# Requires: 3+ running atack pods, a free node for rescheduling.

POD_COUNT=$(kubectl get pods -l app=atack --no-headers | wc -l)
READY_COUNT=$(kubectl get pods -l app=atack --no-headers | grep -c "Running" || true)

if [[ "$POD_COUNT" -lt 2 || "$READY_COUNT" -lt 2 ]]; then
    echo "ERROR: need at least 2 running atack pods, got ${READY_COUNT}/${POD_COUNT}"
    exit 1
fi
echo "${READY_COUNT} pods running"
kubectl get pods -l app=atack -o wide

# Pick a pod to kill (highest index, skip control-plane nodes).
CP_NODES=$(kubectl get nodes -l node-role.kubernetes.io/control-plane \
    -o jsonpath='{.items[*].metadata.name}')

TARGET_POD=""
TARGET_NODE=""
for POD in $(kubectl get pods -l app=atack -o jsonpath='{.items[*].metadata.name}'); do
    NODE=$(kubectl get pod "$POD" -o jsonpath='{.spec.nodeName}')
    IS_CP=false
    for CP in $CP_NODES; do
        [[ "$NODE" == "$CP" ]] && IS_CP=true && break
    done
    if [[ "$IS_CP" == "false" ]]; then
        TARGET_POD="$POD"
        TARGET_NODE="$NODE"
    fi
done

if [[ -z "$TARGET_POD" ]]; then
    echo "ERROR: no suitable pod found"
    exit 1
fi

# Look up the IMEX daemon pod name before killing anything.
IMEX_POD=$(kubectl get pods -n nvidia-dra-driver-gpu \
    --field-selector "spec.nodeName=${TARGET_NODE}" \
    --no-headers 2>/dev/null | grep "computedomain-daemon" | awk '{print $1}' || true)

echo ""
echo "Will taint ${TARGET_NODE} NoSchedule, then force-kill ${TARGET_POD} + ${IMEX_POD:-<no IMEX daemon>}"
echo "Press Enter to proceed, Ctrl+C to abort"
read -r

# Ensure taint is removed on any exit (success, error, Ctrl+C).
cleanup() {
    echo ""
    echo "Removing taint from ${TARGET_NODE}..."
    kubectl taint nodes "${TARGET_NODE}" simulate-failure=true:NoSchedule- 2>/dev/null || true
}
trap cleanup EXIT

T0=$(date +%s)
echo "Tainting ${TARGET_NODE} NoSchedule, force-killing pods..."
set -x
kubectl taint nodes "${TARGET_NODE}" simulate-failure=true:NoSchedule --overwrite
kubectl delete pod "${TARGET_POD}" --grace-period=0 --force &
if [[ -n "$IMEX_POD" ]]; then
    kubectl delete pod -n nvidia-dra-driver-gpu "${IMEX_POD}" --grace-period=0 --force &
fi
wait
set +x
echo "All killed. IMEX daemon: ${IMEX_POD:-<none found>}"

echo ""
echo "Waiting for ALL pods to be Ready with passing liveness..."
# Wait for the killed pod to disappear before checking for the replacement.
echo "Waiting for ${TARGET_POD} to disappear..."
while kubectl get pod "${TARGET_POD}" -o jsonpath='{.spec.nodeName}' 2>/dev/null | grep -q "${TARGET_NODE}"; do
    sleep 1
done
echo "${TARGET_POD} gone from ${TARGET_NODE}"
echo ""
echo "Waiting for all ${POD_COUNT} pods to report fresh error-free results..."

ATTEMPTS=0
while true; do
    ATTEMPTS=$((ATTEMPTS + 1))
    ELAPSED=$(( $(date +%s) - T0 ))

    ALL_OK=true
    REPORTING_COUNT=0
    POD_LINES=""
    # Use jsonpath to avoid awk field-parsing issues (restart column
    # has variable width like "2 (94s ago)").
    while IFS=$'\t' read -r POD READY_FIELD NODE; do
        [[ -z "$POD" ]] && continue

        # Query /results to check for a complete error-free round.
        RESULT_STATUS="?"
        NODE_IP=$(kubectl get node "$NODE" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || echo "")
        if [[ -n "$NODE_IP" && "$READY_FIELD" == "true" ]]; then
            RESULT_JSON=$(curl -s --connect-timeout 1 --max-time 2 "http://${NODE_IP}:1337/results" 2>/dev/null || echo "")
            if [[ -n "$RESULT_JSON" ]]; then
                eval "$(echo "$RESULT_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d.get('results', [])
if not r:
    print('TOTAL=0; ERRORS=-1; AGE=0; NO_RESULTS=yes')
else:
    last = r[-1]
    print(f'TOTAL={last[\"total\"]}; ERRORS={last[\"errors\"]}; AGE={last[\"age_s\"]}; NO_RESULTS=no')
" 2>/dev/null || echo "TOTAL=0; ERRORS=-1; AGE=0; NO_RESULTS=yes")"
                if [[ "$NO_RESULTS" == "yes" ]]; then
                    RESULT_STATUS="no results yet"
                    ALL_OK=false
                elif python3 -c "exit(0 if float('${AGE}') > 10 else 1)"; then
                    RESULT_STATUS="stale(${AGE}s)"
                    ALL_OK=false
                elif [[ "$ERRORS" == "0" && "$TOTAL" != "0" ]]; then
                    RESULT_STATUS="ok(${TOTAL},${AGE}s)"
                    REPORTING_COUNT=$((REPORTING_COUNT + 1))
                elif [[ "$TOTAL" != "0" ]]; then
                    RESULT_STATUS="${ERRORS}err/${TOTAL}(${AGE}s)"
                    ALL_OK=false
                else
                    RESULT_STATUS="no results"
                    ALL_OK=false
                fi
            else
                RESULT_STATUS="unreachable"
                ALL_OK=false
            fi
        else
            ALL_OK=false
        fi

        POD_LINES="${POD_LINES}  ${POD} ${READY_FIELD} ${NODE} results=${RESULT_STATUS}\n"
    done < <(kubectl get pods -l app=atack -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].ready}{"\t"}{.spec.nodeName}{"\n"}{end}' 2>/dev/null)

    echo "  [${ELAPSED}s]"
    echo -e "$POD_LINES"

    if [[ "$ALL_OK" == "true" && "$REPORTING_COUNT" -ge "$POD_COUNT" ]]; then
        NEW_NODE=$(kubectl get pod "${TARGET_POD}" -o jsonpath='{.spec.nodeName}' 2>/dev/null || echo "?")
        echo ""
        if [[ "$NEW_NODE" == "$TARGET_NODE" ]]; then
            echo "WARNING: ${TARGET_POD} rescheduled to the SAME node (${NEW_NODE})"
        else
            echo "All pods producing full error-free results after ${ELAPSED}s"
            echo "${TARGET_POD}: ${TARGET_NODE} → ${NEW_NODE}"
        fi
        break
    fi

    if [[ "$ATTEMPTS" -ge 120 ]]; then
        echo "ERROR: timed out after ${ELAPSED}s"
        kubectl get pods -l app=atack -o wide
        exit 1
    fi
    sleep 3
done

# Taint removed by EXIT trap.
echo ""
kubectl get pods -l app=atack -o wide
