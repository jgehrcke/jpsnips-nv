# Cleanup from previous attempt.
# So one can run `bash 1-node-breaks.sh` repeatedly.
kubectl get resourceclaim
kubectl get computedomains.resource.nvidia.com
kubectl delete -f 1-node-job.yaml
kubectl delete resourceclaim/repro2-compute-domain-shared-channel
kubectl delete computedomains.resource.nvidia.com repro2-compute-domain

# Create CD
cat > cd.yaml <<EOF
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: repro2-compute-domain
spec:
  numNodes: 1
  channel:
    resourceClaimTemplate:
      name: repro2-compute-domain-channel
EOF
kubectl apply -f cd.yaml

# Extract UID
export _CDUID=$(kubectl get computedomain.resource.nvidia.com repro2-compute-domain -o yaml | grep 'uid: ' | awk '{print $2}')
echo "CDUID: ${_CDUID}"

# Create RC, referring to specific CD
cat > rc-for-specific-cd.yaml <<EOF
apiVersion:  resource.k8s.io/v1beta1
kind: ResourceClaim
metadata:
  name: repro2-compute-domain-shared-channel
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
kubectl apply -f 1-node-job.yaml

sleep 12

echo -e "\n\npods on nodes:"
kubectl get pods -o wide

export _DAEMON_POD=$(kubectl get pods -n nvidia-dra-driver-gpu | grep 'repro2-compute-domain-' |  awk '{print $1}')
echo "DAEMON_POD: ${_DAEMON_POD}"

echo -e "\n\nIMEX daemon status:"
kubectl exec -n nvidia-dra-driver-gpu ${_DAEMON_POD} -- nvidia-imex-ctl -q
kubectl exec -n nvidia-dra-driver-gpu ${_DAEMON_POD} -- nvidia-imex-ctl -N
kubectl exec -n nvidia-dra-driver-gpu ${_DAEMON_POD} -- nvidia-imex-ctl -s -t 2000

echo -e "\n\nleader log tail:"
kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=0,job-name=repro2

echo -e "\n\nfollower log tail:"
kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=1,job-name=repro2

echo -e "\n\nIMEX daemon log:"
kubectl logs -l resource.nvidia.com/computeDomain=${_CDUID} -n nvidia-dra-driver-gpu

