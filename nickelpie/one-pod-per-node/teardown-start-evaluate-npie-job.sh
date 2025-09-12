# kudos to
# https://github.com/mattbryson/bash-arg-parse/blob/master/arg_parse_example
# adjusted to export vars.

_script_name=$(basename "$0")

function usage
{
    echo "parameters:"
    echo "  -n | --n-ranks              : todo";
    echo "  -m | --matrix-scale         : todo";
    echo "  -g | --gb-per-benchmark     : todo";
    echo "  -s | --sleep-after-work     : todo";
    echo "  -h | --help                 : This message";
}

function parse_args
{
  # positional args
  args=()

  # named args
  while [ "$1" != "" ]; do
      case "$1" in
          -n | --n-ranks )             export NICKELPIE_N_RANKS="$2"; shift;;
          -m | --matrix-scale )        export NICKELPIE_MATRIX_SCALE="$2"; shift;;
          -g | --gb-per-benchmark )    export NICKELPIE_SEND_TOTAL_GB_PER_BENCHMARK="$2"; shift;;
          -s | --sleep-after-work )    export NICKELPIE_SLEEP_AFTER_WORK="$2"; shift;;
          -h | --help )                usage;                   exit;;
          * )                          args+=("$1")             # if no match, add it to the positional args
      esac
      shift
  done

  # restore positional args
  set -- "${args[@]}"

  # set defaults
  if [[ -z "$NICKELPIE_N_RANKS" ]]; then
      export NICKELPIE_N_RANKS="2"
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
kubectl delete -f npie-job.yaml.rendered

kubectl apply -f npie-job.yaml.rendered

for RANK in $(seq 1 "$((${NICKELPIE_N_RANKS}-1))");
do
    kubectl wait --for=condition=Ready pods -l batch.kubernetes.io/job-completion-index=${RANK},job-name=nickelpie-test --timeout=40s
done

# show node hostnames
kubectl get pods -o wide
sleep 2

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
