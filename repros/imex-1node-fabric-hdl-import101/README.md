## Container image dev

Build, push `latest`:

```console
docker buildx build --progress plain . \
    -t jgehrcke/fiberrepro -f repro.Dockerfile  \
        && docker push jgehrcke/fiberrepro
```

Create, push a tag:

```console
docker tag jgehrcke/fiberrepro jgehrcke/fiberrepro:v250324
docker push jgehrcke/fiberrepro:v250324
```

## Env

- k8s 1.32
- NVIDIA DRA Driver for GPUs `25.3.0-rc.1`
- NVIDIA GPU Operator `25.3.0-rc.2`

```
$ kubectl version
Client Version: v1.32.0
Kustomize Version: v5.5.0
Server Version: v1.32.0
```

```
$ helm list -A
NAME                 	NAMESPACE            	REVISION	UPDATED                                	STATUS  	CHART                            	APP VERSION
gpu-operator         	gpu-operator         	1       	2025-03-17 19:14:32.936849317 +0000 UTC	deployed	gpu-operator-v25.3.0-rc.2        	devel-ubi8
nvidia-dra-driver-gpu	nvidia-dra-driver-gpu	1       	2025-03-21 18:55:01.527997891 +0000 UTC	deployed	nvidia-dra-driver-gpu-25.3.0-rc.1	25.3.0-rc.1
```

## Sample output: 1 node scenario

```console
$ bash 1-node-breaks.sh
NAME                                   STATE                AGE
repro2-compute-domain-shared-channel   allocated,reserved   32s
NAME                    AGE
repro1-compute-domain   10h
repro2-compute-domain   33s
service "svc-repro2" deleted
job.batch "repro2" deleted
resourceclaim.resource.k8s.io "repro2-compute-domain-shared-channel" deleted
computedomain.resource.nvidia.com "repro2-compute-domain" deleted
computedomain.resource.nvidia.com/repro2-compute-domain created
CDUID: 72c69841-b756-43e2-b8a9-3a88a2e5f3f2
resourceclaim.resource.k8s.io/repro2-compute-domain-shared-channel created
service/svc-repro2 created
job.batch/repro2 created


pods on nodes:
NAME             READY   STATUS      RESTARTS   AGE   IP              NODE                   NOMINATED NODE   READINESS GATES
repro2-0-pcnsj   1/1     Running     0          12s   192.168.33.52   sc-starwars-mab8-b00   <none>           <none>
repro2-1-srtz6   0/1     Completed   0          12s   192.168.33.48   sc-starwars-mab8-b00   <none>           <none>
DAEMON_POD: repro2-compute-domain-cm57m-d5hfr


IMEX daemon status:
READY
Connectivity Table Legend:
I - Invalid - Node wasn't reachable, no connection status available
N - Never Connected
R - Recovering - Connection was lost, but clean up has not yet been triggered.
D - Disconnected - Connection was lost, and clean up has been triggreed.
A - Authenticating - If GSSAPI enabled, client has initiated mutual authentication.
C - Connected - Ready for operation

3/25/2025 10:39:43.659
Nodes:
Node #0   - 10.136.206.48   - READY

 Nodes From\To  0
       0        C
Domain State: UP
READY stopAtReady: 0
keepGoing: 1
Finishing subscription
READY


leader log tail:
[pod/repro2-0-pcnsj/repro2]  'uuid': UUID('af0684a1-a407-1553-0ec4-750609f59600')}
[pod/repro2-0-pcnsj/repro2] 250325-10:39:35.272 INFO: driver.cuDeviceGet(0) returned: <CUdevice 0>
[pod/repro2-0-pcnsj/repro2] 250325-10:39:35.272 INFO: cudev: <CUdevice 0> type: <class 'cuda.bindings.driver.CUdevice'>
[pod/repro2-0-pcnsj/repro2] 250325-10:39:35.273 INFO: exportable fabric handle object: data : b'\x00\x00 \x00\x00\x00\x00\x00\x80\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
[pod/repro2-0-pcnsj/repro2] 250325-10:39:35.273 INFO: exportable fabric handle data: 00002000000000008016000000000000000000000000000000000000000000000000000000000000000000000000150000000000000000000000000000000000
[pod/repro2-0-pcnsj/repro2] 250325-10:39:35.273 INFO: start HTTP server
[pod/repro2-0-pcnsj/repro2] 250325-10:39:37.371 INFO: HTTPD: follower requests fabric handle
[pod/repro2-0-pcnsj/repro2] 192.168.33.48 - - [25/Mar/2025 10:39:37] "GET /fabric-handle HTTP/1.1" 200 -
[pod/repro2-0-pcnsj/repro2] 250325-10:39:37.372 INFO: shutdown soon
[pod/repro2-0-pcnsj/repro2] 250325-10:39:37.372 INFO: served request to /fabric-handle


follower log tail:
[pod/repro2-1-srtz6/repro2]     main()
[pod/repro2-1-srtz6/repro2]   File "/thing/fabric-handle-transfer-test.py", line 98, in main
[pod/repro2-1-srtz6/repro2]     follower_raw_cuda_mem_share_over_imex(fabric_handle_data)
[pod/repro2-1-srtz6/repro2]   File "/thing/fabric-handle-transfer-test.py", line 217, in follower_raw_cuda_mem_share_over_imex
[pod/repro2-1-srtz6/repro2]     handle_local = checkCudaErrors(
[pod/repro2-1-srtz6/repro2]                    ^^^^^^^^^^^^^^^^
[pod/repro2-1-srtz6/repro2]   File "/thing/fabric-handle-transfer-test.py", line 343, in checkCudaErrors
[pod/repro2-1-srtz6/repro2]     raise RuntimeError(
[pod/repro2-1-srtz6/repro2] RuntimeError: CUDA error code=101(b'CUDA_ERROR_INVALID_DEVICE')
[pod/repro2-1-srtz6/repro2] 250325-10:39:37.374 INFO: exit 0


IMEX daemon log:
[Mar 25 2025 10:39:32] [INFO] [tid 38] Identified this node as ID 0, using bind IP of '10.136.206.48', and network interface of enP2s2f1
[Mar 25 2025 10:39:32] [INFO] [tid 38] nvidia-imex persistence file /var/run/nvidia-imex/persist.dat does not exist.  Assuming no previous importers.
[Mar 25 2025 10:39:32] [INFO] [tid 38] NvGpu Library version matched with GPU Driver version
[Mar 25 2025 10:39:32] [INFO] [tid 62] Started processing of incoming messages.
[Mar 25 2025 10:39:32] [INFO] [tid 63] Started processing of incoming messages.
[Mar 25 2025 10:39:32] [INFO] [tid 64] Started processing of incoming messages.
[Mar 25 2025 10:39:32] [INFO] [tid 65] Started processing of incoming messages.
[Mar 25 2025 10:39:32] [INFO] [tid 38] Creating gRPC channels to all peers (nPeers = 1).
[Mar 25 2025 10:39:32] [INFO] [tid 38] IMEX_WAIT_FOR_QUORUM != FULL, continuing initialization without waiting for connections to all nodes.
[Mar 25 2025 10:39:32] [INFO] [tid 38] GPU event successfully subscribed
```

## Sample output: 2 nodes scenario

```console
$ bash 2-nodes-works.sh
computedomain.resource.nvidia.com "repro1-compute-domain" deleted
service "svc-repro1" deleted
job.batch "repro1" deleted
computedomain.resource.nvidia.com/repro1-compute-domain created
service/svc-repro1 created
job.batch/repro1 created
NAME             READY   STATUS    RESTARTS   AGE   IP              NODE                   NOMINATED NODE   READINESS GATES
repro1-0-92t7w   1/1     Running   0          15s   192.168.33.59   sc-starwars-mab8-b00   <none>           <none>
repro1-1-9sqd9   1/1     Running   0          15s   192.168.32.44   sc-starwars-mab7-b00   <none>           <none>
DAEMON_POD: repro1-compute-domain-8thkg-n2gv2


IMEX daemon status:
READY
Connectivity Table Legend:
I - Invalid - Node wasn't reachable, no connection status available
N - Never Connected
R - Recovering - Connection was lost, but clean up has not yet been triggered.
D - Disconnected - Connection was lost, and clean up has been triggreed.
A - Authenticating - If GSSAPI enabled, client has initiated mutual authentication.
C - Connected - Ready for operation

3/25/2025 10:42:08.082
Nodes:
Node #0   - 10.136.206.47   - READY
Node #1   - 10.136.206.48   - READY

 Nodes From\To  0  1
       0        C  C
       1        C  C
Domain State: UP
READY stopAtReady: 0
keepGoing: 1
Finishing subscription
READY
[pod/repro1-1-9sqd9/repro1]  'multiProcessorCount': 132,
[pod/repro1-1-9sqd9/repro1]  'name': b'NVIDIA GH200 96GB HBM3',
[pod/repro1-1-9sqd9/repro1]  'pciBusID': 1,
[pod/repro1-1-9sqd9/repro1]  'pciDeviceID': 0,
[pod/repro1-1-9sqd9/repro1]  'pciDomainID': 25,
[pod/repro1-1-9sqd9/repro1]  'uuid': UUID('ea62b7cb-4590-ab4c-ef4d-5e73558e11df')}
[pod/repro1-1-9sqd9/repro1] 250325-10:41:59.113 INFO: enter loop: get fabric handle from leader, URL: http://repro1-0.svc-repro1:1337/fabric-handle
[pod/repro1-1-9sqd9/repro1] 250325-10:41:59.117 INFO: leader responded, response body (hex prefix): 4141416741...
[pod/repro1-1-9sqd9/repro1] 250325-10:41:59.117 INFO: fabric handle data: b'\x00\x00 \x00\x00\x00\x00\x00\x81\x16\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
[pod/repro1-1-9sqd9/repro1] 250325-10:41:59.120 INFO: cuMemImportFromShareableHandle(): successful
[pod/repro1-0-92t7w/repro1]  'uuid': UUID('af0684a1-a407-1553-0ec4-750609f59600')}
[pod/repro1-0-92t7w/repro1] 250325-10:41:59.098 INFO: driver.cuDeviceGet(0) returned: <CUdevice 0>
[pod/repro1-0-92t7w/repro1] 250325-10:41:59.098 INFO: cudev: <CUdevice 0> type: <class 'cuda.bindings.driver.CUdevice'>
[pod/repro1-0-92t7w/repro1] 250325-10:41:59.099 INFO: exportable fabric handle object: data : b'\x00\x00 \x00\x00\x00\x00\x00\x81\x16\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
[pod/repro1-0-92t7w/repro1] 250325-10:41:59.099 INFO: exportable fabric handle data: 00002000000000008116000000000000010000000000000000000000000000000000000000000000000000000000150000000000000000000000000000000000
[pod/repro1-0-92t7w/repro1] 250325-10:41:59.100 INFO: start HTTP server
[pod/repro1-0-92t7w/repro1] 250325-10:41:59.116 INFO: HTTPD: follower requests fabric handle
[pod/repro1-0-92t7w/repro1] 192.168.32.44 - - [25/Mar/2025 10:41:59] "GET /fabric-handle HTTP/1.1" 200 -
[pod/repro1-0-92t7w/repro1] 250325-10:41:59.116 INFO: shutdown soon
[pod/repro1-0-92t7w/repro1] 250325-10:41:59.117 INFO: served request to /fabric-handle


IMEX 1 daemon log:
IMEX Log initializing at: 3/25/2025 10:41:54.950
[Mar 25 2025 10:41:54] [INFO] [tid 38] IMEX version 570.00 is running with the following configuration options

[Mar 25 2025 10:41:54] [INFO] [tid 38] Logging level = 4

[Mar 25 2025 10:41:54] [INFO] [tid 38] Logging file name/path = /var/log/nvidia-imex.log

[Mar 25 2025 10:41:54] [INFO] [tid 38] Append to log file = 0

[Mar 25 2025 10:41:54] [INFO] [tid 38] Max Log file size = 1024 (MBs)

[Mar 25 2025 10:41:54] [INFO] [tid 38] Use Syslog file = 0

[Mar 25 2025 10:41:54] [INFO] [tid 38] IMEX Library communication bind interface =

[Mar 25 2025 10:41:54] [INFO] [tid 38] IMEX library communication bind port = 50000

[Mar 25 2025 10:41:54] [INFO] [tid 38] Identified this node as ID 1, using bind IP of '10.136.206.48', and network interface of enP2s2f1
[Mar 25 2025 10:41:54] [INFO] [tid 38] nvidia-imex persistence file /var/run/nvidia-imex/persist.dat does not exist.  Assuming no previous importers.
[Mar 25 2025 10:41:54] [INFO] [tid 38] NvGpu Library version matched with GPU Driver version
[Mar 25 2025 10:41:54] [INFO] [tid 62] Started processing of incoming messages.
[Mar 25 2025 10:41:54] [INFO] [tid 63] Started processing of incoming messages.
[Mar 25 2025 10:41:54] [INFO] [tid 64] Started processing of incoming messages.
[Mar 25 2025 10:41:54] [INFO] [tid 38] Creating gRPC channels to all peers (nPeers = 2).
[Mar 25 2025 10:41:54] [INFO] [tid 65] Started processing of incoming messages.
[Mar 25 2025 10:41:54] [INFO] [tid 38] IMEX_WAIT_FOR_QUORUM != FULL, continuing initialization without waiting for connections to all nodes.
[Mar 25 2025 10:41:54] [INFO] [tid 38] GPU event successfully subscribed
[Mar 25 2025 10:41:59] [INFO] [tid 61] Attaching to GPU 590080.
[Mar 25 2025 10:41:59] [INFO] [tid 61] Attached to GPU id: 590080, PCI Info: domain - 9, busId - 1, slot - 0, function - 0
[Mar 25 2025 10:41:59] [INFO] [tid 61] GPU id: 590080, UUID: GPU-af0684a1-a4
DAEMON_POD: repro1-compute-domain-8thkg-zfvlh


IMEX 2 daemon log:
IMEX Log initializing at: 3/25/2025 10:41:54.338
[Mar 25 2025 10:41:54] [INFO] [tid 38] IMEX version 570.00 is running with the following configuration options

[Mar 25 2025 10:41:54] [INFO] [tid 38] Logging level = 4

[Mar 25 2025 10:41:54] [INFO] [tid 38] Logging file name/path = /var/log/nvidia-imex.log

[Mar 25 2025 10:41:54] [INFO] [tid 38] Append to log file = 0

[Mar 25 2025 10:41:54] [INFO] [tid 38] Max Log file size = 1024 (MBs)

[Mar 25 2025 10:41:54] [INFO] [tid 38] Use Syslog file = 0

[Mar 25 2025 10:41:54] [INFO] [tid 38] IMEX Library communication bind interface =

[Mar 25 2025 10:41:54] [INFO] [tid 38] IMEX library communication bind port = 50000

[Mar 25 2025 10:41:54] [INFO] [tid 38] Identified this node as ID 0, using bind IP of '10.136.206.47', and network interface of enP2s2f1
[Mar 25 2025 10:41:54] [INFO] [tid 38] nvidia-imex persistence file /var/run/nvidia-imex/persist.dat does not exist.  Assuming no previous importers.
[Mar 25 2025 10:41:54] [INFO] [tid 38] NvGpu Library version matched with GPU Driver version
[Mar 25 2025 10:41:54] [INFO] [tid 63] Started processing of incoming messages.
[Mar 25 2025 10:41:54] [INFO] [tid 62] Started processing of incoming messages.
[Mar 25 2025 10:41:54] [INFO] [tid 64] Started processing of incoming messages.
[Mar 25 2025 10:41:54] [INFO] [tid 38] Creating gRPC channels to all peers (nPeers = 2).
[Mar 25 2025 10:41:54] [INFO] [tid 65] Started processing of incoming messages.
[Mar 25 2025 10:41:54] [INFO] [tid 38] IMEX_WAIT_FOR_QUORUM != FULL, continuing initialization without waiting for connections to all nodes.
[Mar 25 2025 10:41:54] [INFO] [tid 38] GPU event successfully subscribed
```
