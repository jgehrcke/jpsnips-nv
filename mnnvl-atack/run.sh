#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <num_pods> <chunk_mib>"
    echo "  num_pods:  number of StatefulSet replicas (one per node)"
    echo "  chunk_mib: GPU memory chunk size in MiB per peer-round"
    exit 1
fi

export REPLICAS="$1"
export CHUNK_MIB="$2"
export POLL_INTERVAL_S="3"

echo "--- Cleaning up previous resources (if any)"
kubectl delete statefulset atack --ignore-not-found
kubectl delete service svc-atack --ignore-not-found
kubectl delete computedomain atack-compute-domain --ignore-not-found
# Wait for pods to terminate before redeploying.
kubectl wait --for=delete pod -l app=atack --timeout=60s 2>/dev/null || true

echo "--- Rendering manifest: REPLICAS=${REPLICAS}, CHUNK_MIB=${CHUNK_MIB}, POLL_INTERVAL_S=${POLL_INTERVAL_S}"
RENDERED=$(envsubst '${REPLICAS} ${CHUNK_MIB} ${POLL_INTERVAL_S}' < "${SCRIPT_DIR}/atack.yaml.envsubst")

echo "--- Applying manifest"
echo "$RENDERED" | kubectl apply -f -

echo "--- Waiting for pods to become ready"
kubectl rollout status statefulset/atack --timeout=300s

echo "--- Pod status"
kubectl get pods -l app=atack -o wide
