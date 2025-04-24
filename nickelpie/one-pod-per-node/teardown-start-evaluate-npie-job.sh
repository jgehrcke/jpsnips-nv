

if [ $# -ne 1 ]; then
    echo "first arg: required: number of nodes/ranks"
    exit 1
else
    export NICKELPIE_N_RANKS="$1"
    echo "using NICKELPIE_N_RANKS: ${NICKELPIE_N_RANKS}"
fi

cat npie-job.yaml | envsubst > npie-job.yaml.rendered


set -x
kubectl get resourceclaim
kubectl get computedomains.resource.nvidia.com
kubectl delete -f npie-job.yaml.rendered
kubectl delete computedomains.resource.nvidia.com nickelpie-test-compute-domain

kubectl apply -f npie-job.yaml.rendered


kubectl get resourceclaims.resource.k8s.io

for RANK in $(seq 1 "$((${NICKELPIE_N_RANKS}-1))");
do
    kubectl wait --for=condition=Ready pods -l batch.kubernetes.io/job-completion-index=${RANK},job-name=nickelpie-test --timeout=40s
done


kubectl get resourceclaims.resource.k8s.io
# How were the RCs constructed? Let's inspect their anatomy.
kubectl describe resourceclaims.resource.k8s.io $(kubectl get resourceclaims.resource.k8s.io | grep nickelpie-test | head -n1 | awk '{print $1}')
kubectl describe resourceclaims.resource.k8s.io $(kubectl get resourceclaims.resource.k8s.io | grep nickelpie-test | tail -n1 | awk '{print $1}')

kubectl get resourceclaimtemplates.resource.k8s.io
kubectl describe resourceclaimtemplates.resource.k8s.io $(kubectl get resourceclaimtemplates.resource.k8s.io | grep nickelpie-test | head -n1 | awk '{print $1}')


# show node hostnames
kubectl get pods -o wide
sleep 2

set +x
echo -e "\n\nleader log tail:"
kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=0,job-name=nickelpie-test

echo -e "\n\nfollower 1 log tail:"
kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=1,job-name=nickelpie-test

# Wait for entire job (all pods) to have completed
set -x
kubectl wait --for=condition=complete --timeout=30s job/nickelpie-test
kubectl wait --for=condition=complete --timeout=30s job/nickelpie-test
set +x

echo -e "\n\nleader result:"
kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=0,job-name=nickelpie-test --tail=-1 | grep RESULT

for RANK in $(seq 1 "$((${NICKELPIE_N_RANKS}-1))");
do
    echo -e "\n\nfollower $RANK result:"
    kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=${RANK},job-name=nickelpie-test --tail=-1 | grep RESULT
done
