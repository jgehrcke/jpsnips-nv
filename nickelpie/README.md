

## usage

```
$ git clone https://github.com/jgehrcke/jpsnips-nv
$ cd jpsnips-nv/nickelpie/one-pod-per-node/
$ bash teardown-start-evaluate-npie-job.sh 2
...
```


## dev: build and push

```
$ docker buildx build \
    --progress plain . -t jgehrcke/nickelpie -f nickelpie.Dockerfile  && \
    docker push jgehrcke/nickelpie

...

#35 DONE 0.0s
Using default tag: latest
The push refers to repository [docker.io/jgehrcke/nickelpie]
```


I tried hard-ish to make this be not-so-bulky, but it's still 1 GB:
```
$ docker image ls | grep nickel | head -n 5
jgehrcke/nickelpie                  latest                             94f4290a7a6f   12 minutes ago   1.07GB
```


## log on GB200

```
$ bash teardown-start-evaluate-npie-job.sh 4
using NICKELPIE_N_RANKS: 4
<snip>
+ kubectl apply -f npie-job.yaml.rendered
computedomain.resource.nvidia.com/nickelpie-test-compute-domain created
service/svc-nickelpie created
job.batch/nickelpie-test created
<snip>
+ kubectl wait --for=condition=Ready pods -l batch.kubernetes.io/job-completion-index=1,job-name=nickelpie-test --timeout=40s
pod/nickelpie-test-1-xfhw9 condition met
<snip>
+ kubectl wait --for=condition=Ready pods -l batch.kubernetes.io/job-completion-index=3,job-name=nickelpie-test --timeout=40s
pod/nickelpie-test-3-6tnkd condition met
+ kubectl get resourceclaims.resource.k8s.io
NAME                                                  STATE                AGE
nickelpie-test-0-dk29f-compute-domain-channel-7bkqv   allocated,reserved   21s
nickelpie-test-1-xfhw9-compute-domain-channel-gtv98   allocated,reserved   21s
nickelpie-test-2-xv6dg-compute-domain-channel-jwqch   allocated,reserved   21s
nickelpie-test-3-6tnkd-compute-domain-channel-vljpj   allocated,reserved   21s
<snip>
+ kubectl describe resourceclaims.resource.k8s.io nickelpie-test-3-6tnkd-compute-domain-channel-vljpj
<snip>
+ kubectl get pods -o wide
NAME                     READY   STATUS    RESTARTS   AGE   IP                NODE                  NOMINATED NODE   READINESS GATES
nickelpie-test-0-dk29f   1/1     Running   0          21s   192.168.154.73    gb-nvl-043-bianca-4   <none>           <none>
nickelpie-test-1-xfhw9   1/1     Running   0          21s   192.168.94.49     gb-nvl-043-bianca-2   <none>           <none>
nickelpie-test-2-xv6dg   1/1     Running   0          21s   192.168.219.146   gb-nvl-043-bianca09   <none>           <none>
nickelpie-test-3-6tnkd   1/1     Running   0          21s   192.168.85.2      gb-nvl-043-bianca-1   <none>           <none>
+ sleep 2
+ set +x


leader log tail:
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:31.064 INFO 265711990403456: HTTPD: follower reached barrier BENCHMARK_1, block emitting response until leader ready
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:31.064 INFO 265711990403456: HTTPD: leader reached barrier BENCHMARK_1. Notify follower to move past this (via HTTP response)
[pod/nickelpie-test-0-dk29f/nickelpie-test] 192.168.94.49 - - [24/Apr/2025 12:08:31] "GET /sync-on-BENCHMARK_1 HTTP/1.1" 200 -
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:31.064 INFO 265715295667424: sync: follower reached barrier BENCHMARK_1 -- move on
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:31.064 INFO 265715295667424: start: send()-to-1: 408 reps
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:31.067 INFO 265715295667424: done: send()-to-1: 408 reps, took 0.002676 s (started at 12:08:31.064460)
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:31.067 INFO 265715295667424: send() loop: done (all calls emitted, maybe not yet done on GPU)
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:31.067 INFO 265715295667424: synchronize with GPU: wait for last send() to have completed
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:31.289 INFO 265711990403456: HTTPD: follower reached barrier WARMUP_EXCHANGE_3, block emitting response until leader ready
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:31.289 INFO 265712044077440: HTTPD: follower reached barrier WARMUP_EXCHANGE_2, block emitting response until leader ready


follower 1 log tail:
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:29.743 INFO 256645679039712: start: communicator.recv() 405/408
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:29.743 INFO 256645679039712: done: communicator.recv() 405/408, took 0.000012 s (started at 12:08:29.743145)
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:29.743 INFO 256645679039712: start: communicator.recv() 406/408
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:29.743 INFO 256645679039712: done: communicator.recv() 406/408, took 0.000013 s (started at 12:08:29.743183)
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:29.743 INFO 256645679039712: start: communicator.recv() 407/408
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:29.743 INFO 256645679039712: done: communicator.recv() 407/408, took 0.000013 s (started at 12:08:29.743223)
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:29.743 INFO 256645679039712: start: communicator.recv() 408/408
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:29.743 INFO 256645679039712: done: communicator.recv() 408/408, took 0.000015 s (started at 12:08:29.743263)
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:29.743 INFO 256645679039712: recv() loop: done (all calls emitted, maybe not yet done on GPU)
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:29.743 INFO 256645679039712: synchronize with GPU: wait for last recv() to have completed
+ kubectl wait --for=condition=complete --timeout=30s job/nickelpie-test
job.batch/nickelpie-test condition met
+ kubectl wait --for=condition=complete --timeout=30s job/nickelpie-test
job.batch/nickelpie-test condition met
+ set +x


leader result:
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:33.936 INFO 265715295667424: send-recv-0->1(0) RESULT data sent: 1999.200 GB, time elapsed: 2.871 s
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:33.936 INFO 265715295667424: send-recv-0->1(0) RESULT bandwidth: 696.245 GB/s
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:37.011 INFO 265715295667424: send-recv-0->2(0) RESULT data sent: 1999.200 GB, time elapsed: 2.872 s
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:37.011 INFO 265715295667424: send-recv-0->2(0) RESULT bandwidth: 695.987 GB/s
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:40.085 INFO 265715295667424: send-recv-0->3(0) RESULT data sent: 1999.200 GB, time elapsed: 2.868 s
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:40.085 INFO 265715295667424: send-recv-0->3(0) RESULT bandwidth: 696.956 GB/s
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:44.532 INFO 265715295667424: broadcast-0->[1, 2, 3](0) RESULT data sent: 5997.600 GB, time elapsed: 4.446 s
[pod/nickelpie-test-0-dk29f/nickelpie-test] 250424-12:08:44.532 INFO 265715295667424: broadcast-0->[1, 2, 3](0) RESULT bandwidth: 1348.973 GB/s


follower 1 result:
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:32.597 INFO 256645679039712: send-recv-0->1(1) RESULT data sent: 1999.200 GB, time elapsed: 2.871 s
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:32.597 INFO 256645679039712: send-recv-0->1(1) RESULT bandwidth: 696.379 GB/s
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:43.194 INFO 256645679039712: broadcast-0->[1, 2, 3](1) RESULT data sent: 5997.600 GB, time elapsed: 4.445 s
[pod/nickelpie-test-1-xfhw9/nickelpie-test] 250424-12:08:43.194 INFO 256645679039712: broadcast-0->[1, 2, 3](1) RESULT bandwidth: 1349.213 GB/s


follower 2 result:
[pod/nickelpie-test-2-xv6dg/nickelpie-test] 250424-12:08:35.673 INFO 251571477955808: send-recv-0->2(2) RESULT data sent: 1999.200 GB, time elapsed: 2.872 s
[pod/nickelpie-test-2-xv6dg/nickelpie-test] 250424-12:08:35.673 INFO 251571477955808: send-recv-0->2(2) RESULT bandwidth: 696.170 GB/s
[pod/nickelpie-test-2-xv6dg/nickelpie-test] 250424-12:08:43.194 INFO 251571477955808: broadcast-0->[1, 2, 3](2) RESULT data sent: 5997.600 GB, time elapsed: 4.446 s
[pod/nickelpie-test-2-xv6dg/nickelpie-test] 250424-12:08:43.194 INFO 251571477955808: broadcast-0->[1, 2, 3](2) RESULT bandwidth: 1349.119 GB/s


follower 3 result:
[pod/nickelpie-test-3-6tnkd/nickelpie-test] 250424-12:08:38.746 INFO 248174890665184: send-recv-0->3(3) RESULT data sent: 1999.200 GB, time elapsed: 2.868 s
[pod/nickelpie-test-3-6tnkd/nickelpie-test] 250424-12:08:38.746 INFO 248174890665184: send-recv-0->3(3) RESULT bandwidth: 697.085 GB/s
[pod/nickelpie-test-3-6tnkd/nickelpie-test] 250424-12:08:43.193 INFO 248174890665184: broadcast-0->[1, 2, 3](3) RESULT data sent: 5997.600 GB, time elapsed: 4.445 s
[pod/nickelpie-test-3-6tnkd/nickelpie-test] 250424-12:08:43.193 INFO 248174890665184: broadcast-0->[1, 2, 3](3) RESULT bandwidth: 1349.258 GB/s
```