kubectl delete -f 2-nodes-job.yaml
sleep 1
kubectl apply -f 2-nodes-job.yaml
sleep 15
kubectl get pods -o wide

# Expect two pods, inspect the first one
export _DAEMON_POD=$(kubectl get pods -n nvidia-dra-driver-gpu | grep 'repro1-compute-domain-' |  head -n1 | awk '{print $1}')
echo "DAEMON_POD: ${_DAEMON_POD}"

echo -e "\n\nIMEX daemon status:"
kubectl exec -n nvidia-dra-driver-gpu ${_DAEMON_POD} -- nvidia-imex-ctl -q
kubectl exec -n nvidia-dra-driver-gpu ${_DAEMON_POD} -- nvidia-imex-ctl -N
kubectl exec -n nvidia-dra-driver-gpu ${_DAEMON_POD} -- nvidia-imex-ctl -s -t 2000

# Show log tail of both pods, prefixed with pod name.
kubectl logs --prefix -l job-name=repro1

echo -e "\n\nIMEX 1 daemon log:"
kubectl logs -n nvidia-dra-driver-gpu ${_DAEMON_POD} -n nvidia-dra-driver-gpu

# Inspect the other pod (is ordering predictable?)
export _DAEMON_POD=$(kubectl get pods -n nvidia-dra-driver-gpu | grep 'repro1-compute-domain-' |  tail -n1 | awk '{print $1}')
echo "DAEMON_POD: ${_DAEMON_POD}"

echo -e "\n\nIMEX 2 daemon log:"
kubectl logs -n nvidia-dra-driver-gpu ${_DAEMON_POD} -n nvidia-dra-driver-gpu