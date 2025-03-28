# Cleanup from previous attempt.
# So one can run `bash 1-node-cd-workaround.sh` repeatedly.
kubectl get resourceclaim
kubectl get computedomains.resource.nvidia.com
kubectl delete -f 1-node-cd-workaround.sh
kubectl delete resourceclaim/cd1node-compute-domain-shared-channel
kubectl delete computedomains.resource.nvidia.com cd1node-compute-domain

# Create ComputeDomain manually.
cat > cd.yaml <<EOF
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: cd1node-compute-domain
spec:
  numNodes: 1
  channel:
    resourceClaimTemplate:
      name: cd1node-compute-domain-channel
EOF
kubectl apply -f cd.yaml

# Extract UID of ComputeDomain.
export _CDUID=$(kubectl get computedomain.resource.nvidia.com cd1node-compute-domain -o yaml | grep 'uid: ' | awk '{print $2}')
echo "CDUID: ${_CDUID}"

# Create ResourceClaim manually, referring to specific ComputeDomain.
cat > rc-for-specific-cd.yaml <<EOF
apiVersion:  resource.k8s.io/v1beta1
kind: ResourceClaim
metadata:
  name: cd1node-compute-domain-shared-channel
spec:
  devices:
    requests:
    - deviceClassName: compute-domain-default-channel.nvidia.com
      name: channel
    config:
    - opaque:
        driver: compute-domain.nvidia.com
        parameters:
          apiVersion: resource.nvidia.com/v1beta1
          domainID: ${_CDUID}
          kind: ComputeDomainChannelConfig
      requests:
      - channel
EOF
kubectl apply -f rc-for-specific-cd.yaml

# Submit 2-pod job, using above's ResourceClaim
kubectl apply -f 1-node-job.yaml

# Inspect pods, and their node placement.
sleep 10
echo -e "\n\npods on nodes:"
kubectl get pods -o wide

# Look up pod running IMEX daemon.
export _DAEMON_POD=$(kubectl get pods -n nvidia-dra-driver-gpu | grep 'cd1node-compute-domain-' |  awk '{print $1}')
echo "DAEMON_POD: ${_DAEMON_POD}"

echo -e "\n\nIMEX daemon status:"
kubectl exec -n nvidia-dra-driver-gpu ${_DAEMON_POD} -- nvidia-imex-ctl -q
kubectl exec -n nvidia-dra-driver-gpu ${_DAEMON_POD} -- nvidia-imex-ctl -N
kubectl exec -n nvidia-dra-driver-gpu ${_DAEMON_POD} -- nvidia-imex-ctl -s -t 2000

echo -e "\n\nleader log tail:"
kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=0,job-name=cd1node

echo -e "\n\nfollower log tail:"
kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=1,job-name=cd1node

echo -e "\n\nIMEX daemon log:"
kubectl logs -l resource.nvidia.com/computeDomain=${_CDUID} -n nvidia-dra-driver-gpu

