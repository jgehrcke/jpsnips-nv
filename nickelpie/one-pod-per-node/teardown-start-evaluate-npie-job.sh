# kudos to
# https://github.com/mattbryson/bash-arg-parse/blob/master/arg_parse_example
# adjusted to _export_ vars.

_script_name=$(basename "$0")

function usage
{
    #echo "usage: ${_script_name} -a AN_ARG -s SOME_MORE_ARGS [-y YET_MORE_ARGS || -h]"
    #echo "   ";
    echo "parameters:"
    echo "  -a | --nodes             : A super special argument";
    echo "  -m | --    : Another argument";
    echo "  -y | --yet_more_args     : An optional argument";
    echo "  -h | --help              : This message";
}

function parse_args
{
  # positional args
  args=()

  # named args
  while [ "$1" != "" ]; do
      case "$1" in
          -n | --n-ranks )             export NICKELPIE_N_RANKS="$2";             shift;;
          -m | --matrix-scale )        export NICKELPIE_MATRIX_SCALE="$2";     shift;;
          -g | --gb-per-benchmark )    export NICKELPIE_SEND_TOTAL_GB_PER_BENCHMARK="$2";      shift;;
          -s | --sleep-after-work )    export NICKELPIE_SLEEP_AFTER_WORK="$2";      shift;;
          -h | --help )                usage;                   exit;; # quit and show usage
          * )                          args+=("$1")             # if no match, add it to the positional args
      esac
      shift # move to next kv pair
  done

  # restore positional args
  set -- "${args[@]}"

  # set positionals to vars
  #positional_1="${args[0]}"
  #positional_2="${args[1]}"

  # validate required args
#   if [[ -z "${an_arg}" || -z "${some_more_args}" ]]; then
#       echo "Invalid arguments"
#       usage
#       exit;
#   fi

  # set defaults
  if [[ -z "$NICKELPIE_N_RANKS" ]]; then
      export NICKELPIE_N_RANKS="3"
  fi

  if [[ -z "$NICKELPIE_MATRIX_SCALE" ]]; then
      export NICKELPIE_MATRIX_SCALE="4.0"
  fi

  if [[ -z "$NICKELPIE_SEND_TOTAL_GB_PER_BENCHMARK" ]]; then
      export NICKELPIE_SEND_TOTAL_GB_PER_BENCHMARK="1200"
  fi

  if [[ -z "$NICKELPIE_SLEEP_AFTER_WORK" ]]; then
      export NICKELPIE_SLEEP_AFTER_WORK="0"
  fi

}

parse_args "$@"

echo "NICKELPIE_N_RANKS: $NICKELPIE_N_RANKS"
echo "NICKELPIE_MATRIX_SCALE: $NICKELPIE_MATRIX_SCALE"
echo "NICKELPIE_SEND_TOTAL_GB_PER_BENCHMARK: $NICKELPIE_SEND_TOTAL_GB_PER_BENCHMARK"
echo "NICKELPIE_SLEEP_AFTER_WORK: $NICKELPIE_SLEEP_AFTER_WORK"

cat npie-job.yaml | envsubst > npie-job.yaml.rendered

set -x
# kubectl get resourceclaim
# kubectl get computedomains.resource.nvidia.com
kubectl delete -f npie-job.yaml.rendered
kubectl delete computedomains.resource.nvidia.com nickelpie-test-compute-domain
# wait for any previous job to complete

kubectl wait --for=condition=complete --timeout=30s job/nickelpie-test
kubectl apply -f npie-job.yaml.rendered

# kubectl get resourceclaims.resource.k8s.io

for RANK in $(seq 1 "$((${NICKELPIE_N_RANKS}-1))");
do
    kubectl wait --for=condition=Ready pods -l batch.kubernetes.io/job-completion-index=${RANK},job-name=nickelpie-test --timeout=40s
done

# How were the RCs constructed? Let's inspect their anatomy.
#kubectl get resourceclaims.resource.k8s.io
#kubectl describe resourceclaims.resource.k8s.io $(kubectl get resourceclaims.resource.k8s.io | grep nickelpie-test | head -n1 | awk '{print $1}')
#kubectl describe resourceclaims.resource.k8s.io $(kubectl get resourceclaims.resource.k8s.io | grep nickelpie-test | tail -n1 | awk '{print $1}')
#kubectl get resourceclaimtemplates.resource.k8s.io
#kubectl describe resourceclaimtemplates.resource.k8s.io $(kubectl get resourceclaimtemplates.resource.k8s.io | grep nickelpie-test | head -n1 | awk '{print $1}')


# show node hostnames
kubectl get pods -o wide
sleep 5

set +x
echo -e "\n\nleader (producer) log tail:"
kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=0,job-name=nickelpie-test

echo -e "\n\nfollower (consumer) 1 log tail:"
kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=1,job-name=nickelpie-test

# Wait for entire job (all pods) to have completed
set -x
kubectl wait --for=condition=complete --timeout=30s job/nickelpie-test
kubectl wait --for=condition=complete --timeout=30s job/nickelpie-test
set +x

echo -e "\n\nleader (producer) perspective:"
kubectl logs --prefix -l batch.kubernetes.io/job-completion-index=0,job-name=nickelpie-test --tail=-1 | grep RESULT

echo -e "\n\nall pods, and their perspective:"
kubectl logs --prefix -l job-name=nickelpie-test --tail=-1 | grep RESULT

echo -e "\n\noverall job status:"
kubectl get job nickelpie-test