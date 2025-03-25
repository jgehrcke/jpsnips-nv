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

## Full container logs

```
$ kubectl logs repro1-0-hrwht
250325-10:54:58.903 INFO: cuDriverGetVersion(): 12080
250325-10:54:58.903 INFO: getLocalRuntimeVersion(): 12080
250325-10:54:58.903 INFO: env: NVIDIA_REQUIRE_CUDA: cuda>=12.8 brand=unknown,driver>=470,driver<471 brand=grid,driver>=470,driver<471 brand=tesla,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=vapps,driver>=470,driver<471 brand=vpc,driver>=470,driver<471 brand=vcs,driver>=470,driver<471 brand=vws,driver>=470,driver<471 brand=cloudgaming,driver>=470,driver<471 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=560,driver<561 brand=grid,driver>=560,driver<561 brand=tesla,driver>=560,driver<561 brand=nvidia,driver>=560,driver<561 brand=quadro,driver>=560,driver<561 brand=quadrortx,driver>=560,driver<561 brand=nvidiartx,driver>=560,driver<561 brand=vapps,driver>=560,driver<561 brand=vpc,driver>=560,driver<561 brand=vcs,driver>=560,driver<561 brand=vws,driver>=560,driver<561 brand=cloudgaming,driver>=560,driver<561 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566
250325-10:54:58.903 INFO: env: NV_CUDA_CUDART_VERSION: 12.8.90-1
250325-10:54:58.903 INFO: env: CUDA_VERSION: 12.8.1
250325-10:54:58.903 INFO: env: NVIDIA_VISIBLE_DEVICES: GPU-bb7e88bb-0b80-f35c-992d-73b620b6ac4e
250325-10:54:58.903 INFO: env: NVIDIA_DRIVER_CAPABILITIES: compute,utility
250325-10:54:58.903 INFO: env: NV_CUDA_LIB_VERSION: 12.8.1-1
250325-10:54:58.903 INFO: env: NVIDIA_PRODUCT_NAME: CUDA
250325-10:54:58.903 INFO: listdir(/dev/nvidia-caps-imex-channels): ['channel0']
250325-10:54:58.903 INFO: /proc/devices contains IMEX devices: ['509 nvidia-caps-imex-channels']
250325-10:54:58.903 INFO: k8s job completion index: 0
250325-10:54:59.063 INFO: getDeviceCount(): 1
250325-10:54:59.066 INFO: device 0 properties:
{'major': 9,
 'minor': 0,
 'multiGpuBoardGroupID': 0,
 'multiProcessorCount': 132,
 'name': b'NVIDIA GH200 96GB HBM3',
 'pciBusID': 1,
 'pciDeviceID': 0,
 'pciDomainID': 9,
 'uuid': UUID('bb7e88bb-0b80-f35c-992d-73b620b6ac4e')}
250325-10:54:59.066 INFO: driver.cuDeviceGet(0) returned: <CUdevice 0>
250325-10:54:59.066 INFO: cudev: <CUdevice 0> type: <class 'cuda.bindings.driver.CUdevice'>
250325-10:54:59.066 INFO: exportable fabric handle object: data : b'\x00\x00 \x00\x00\x00\x00\x00\xd7\x1a\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
250325-10:54:59.066 INFO: exportable fabric handle data: 0000200000000000d71a000000000000010000000000000000000000000000000000000000000000000000000000150000000000000000000000000000000000
250325-10:54:59.067 INFO: start HTTP server
250325-10:55:02.308 INFO: HTTPD: follower requests fabric handle
192.168.33.69 - - [25/Mar/2025 10:55:02] "GET /fabric-handle HTTP/1.1" 200 -
250325-10:55:02.308 INFO: shutdown soon
250325-10:55:02.308 INFO: served request to /fabric-handle
250325-10:55:42.308 INFO: shutdown
```

```
$ kubectl logs repro1-1-288wd
250325-10:54:59.090 INFO: cuDriverGetVersion(): 12080
250325-10:54:59.090 INFO: getLocalRuntimeVersion(): 12080
250325-10:54:59.090 INFO: env: NVIDIA_REQUIRE_CUDA: cuda>=12.8 brand=unknown,driver>=470,driver<471 brand=grid,driver>=470,driver<471 brand=tesla,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=vapps,driver>=470,driver<471 brand=vpc,driver>=470,driver<471 brand=vcs,driver>=470,driver<471 brand=vws,driver>=470,driver<471 brand=cloudgaming,driver>=470,driver<471 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=560,driver<561 brand=grid,driver>=560,driver<561 brand=tesla,driver>=560,driver<561 brand=nvidia,driver>=560,driver<561 brand=quadro,driver>=560,driver<561 brand=quadrortx,driver>=560,driver<561 brand=nvidiartx,driver>=560,driver<561 brand=vapps,driver>=560,driver<561 brand=vpc,driver>=560,driver<561 brand=vcs,driver>=560,driver<561 brand=vws,driver>=560,driver<561 brand=cloudgaming,driver>=560,driver<561 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566
250325-10:54:59.090 INFO: env: NV_CUDA_CUDART_VERSION: 12.8.90-1
250325-10:54:59.090 INFO: env: CUDA_VERSION: 12.8.1
250325-10:54:59.090 INFO: env: NVIDIA_VISIBLE_DEVICES: GPU-0abb8bdb-b6c4-73fc-2b1d-a52db1c353e2
250325-10:54:59.090 INFO: env: NVIDIA_DRIVER_CAPABILITIES: compute,utility
250325-10:54:59.090 INFO: env: NV_CUDA_LIB_VERSION: 12.8.1-1
250325-10:54:59.090 INFO: env: NVIDIA_PRODUCT_NAME: CUDA
250325-10:54:59.090 INFO: listdir(/dev/nvidia-caps-imex-channels): ['channel0']
250325-10:54:59.090 INFO: /proc/devices contains IMEX devices: ['510 nvidia-caps-imex-channels']
250325-10:54:59.090 INFO: k8s job completion index: 1
250325-10:54:59.249 INFO: getDeviceCount(): 1
250325-10:54:59.252 INFO: device 0 properties:
{'major': 9,
 'minor': 0,
 'multiGpuBoardGroupID': 0,
 'multiProcessorCount': 132,
 'name': b'NVIDIA GH200 96GB HBM3',
 'pciBusID': 1,
 'pciDeviceID': 0,
 'pciDomainID': 9,
 'uuid': UUID('0abb8bdb-b6c4-73fc-2b1d-a52db1c353e2')}
250325-10:54:59.252 INFO: enter loop: get fabric handle from leader, URL: http://repro1-0.svc-repro1:1337/fabric-handle
250325-10:54:59.301 INFO: request failed, retry soon -- err was: HTTPConnectionPool(host='repro1-0.svc-repro1', port=1337): Max retries exceeded with url: /fabric-handle (Caused by NameResolutionError("<urllib3.connection.HTTPConnection object at 0xe14001cef4d0>: Failed to resolve 'repro1-0.svc-repro1' ([Errno -2] Name or service not known)"))
250325-10:55:02.305 INFO: leader responded, response body (hex prefix): 4141416741...
250325-10:55:02.305 INFO: fabric handle data: b'\x00\x00 \x00\x00\x00\x00\x00\xd7\x1a\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
250325-10:55:02.308 INFO: cuMemImportFromShareableHandle(): successful
250325-10:55:42.308 INFO: shutdown
```

```
$ kubectl logs repro2-0-4hp8f
250325-10:54:30.734 INFO: cuDriverGetVersion(): 12080
250325-10:54:30.734 INFO: getLocalRuntimeVersion(): 12080
250325-10:54:30.734 INFO: env: NVIDIA_REQUIRE_CUDA: cuda>=12.8 brand=unknown,driver>=470,driver<471 brand=grid,driver>=470,driver<471 brand=tesla,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=vapps,driver>=470,driver<471 brand=vpc,driver>=470,driver<471 brand=vcs,driver>=470,driver<471 brand=vws,driver>=470,driver<471 brand=cloudgaming,driver>=470,driver<471 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=560,driver<561 brand=grid,driver>=560,driver<561 brand=tesla,driver>=560,driver<561 brand=nvidia,driver>=560,driver<561 brand=quadro,driver>=560,driver<561 brand=quadrortx,driver>=560,driver<561 brand=nvidiartx,driver>=560,driver<561 brand=vapps,driver>=560,driver<561 brand=vpc,driver>=560,driver<561 brand=vcs,driver>=560,driver<561 brand=vws,driver>=560,driver<561 brand=cloudgaming,driver>=560,driver<561 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566
250325-10:54:30.734 INFO: env: NV_CUDA_CUDART_VERSION: 12.8.90-1
250325-10:54:30.734 INFO: env: CUDA_VERSION: 12.8.1
250325-10:54:30.734 INFO: env: NVIDIA_VISIBLE_DEVICES: GPU-ac687854-a9a8-1692-6388-97238a7e6923
250325-10:54:30.734 INFO: env: NVIDIA_DRIVER_CAPABILITIES: compute,utility
250325-10:54:30.734 INFO: env: NV_CUDA_LIB_VERSION: 12.8.1-1
250325-10:54:30.734 INFO: env: NVIDIA_PRODUCT_NAME: CUDA
250325-10:54:30.734 INFO: listdir(/dev/nvidia-caps-imex-channels): ['channel0']
250325-10:54:30.734 INFO: /proc/devices contains IMEX devices: ['508 nvidia-caps-imex-channels']
250325-10:54:30.734 INFO: k8s job completion index: 0
250325-10:54:30.923 INFO: getDeviceCount(): 1
250325-10:54:30.926 INFO: device 0 properties:
{'major': 9,
 'minor': 0,
 'multiGpuBoardGroupID': 0,
 'multiProcessorCount': 132,
 'name': b'NVIDIA GH200 96GB HBM3',
 'pciBusID': 1,
 'pciDeviceID': 0,
 'pciDomainID': 25,
 'uuid': UUID('ac687854-a9a8-1692-6388-97238a7e6923')}
250325-10:54:30.926 INFO: driver.cuDeviceGet(0) returned: <CUdevice 0>
250325-10:54:30.926 INFO: cudev: <CUdevice 0> type: <class 'cuda.bindings.driver.CUdevice'>
250325-10:54:30.926 INFO: exportable fabric handle object: data : b'\x00\x00 \x00\x00\x00\x00\x00\x82\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
250325-10:54:30.926 INFO: exportable fabric handle data: 00002000000000008216000000000000000000000000000000000000000000000000000000000000000000000000150000000000000000000000000000000000
250325-10:54:30.927 INFO: start HTTP server
250325-10:54:33.045 INFO: HTTPD: follower requests fabric handle
192.168.33.3 - - [25/Mar/2025 10:54:33] "GET /fabric-handle HTTP/1.1" 200 -
250325-10:54:33.046 INFO: shutdown soon
250325-10:54:33.046 INFO: served request to /fabric-handle
250325-10:55:13.046 INFO: shutdown
```

```
$ kubectl logs repro2-1-7mgjd
250325-10:54:29.877 INFO: cuDriverGetVersion(): 12080
250325-10:54:29.877 INFO: getLocalRuntimeVersion(): 12080
250325-10:54:29.877 INFO: env: NVIDIA_REQUIRE_CUDA: cuda>=12.8 brand=unknown,driver>=470,driver<471 brand=grid,driver>=470,driver<471 brand=tesla,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=vapps,driver>=470,driver<471 brand=vpc,driver>=470,driver<471 brand=vcs,driver>=470,driver<471 brand=vws,driver>=470,driver<471 brand=cloudgaming,driver>=470,driver<471 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=560,driver<561 brand=grid,driver>=560,driver<561 brand=tesla,driver>=560,driver<561 brand=nvidia,driver>=560,driver<561 brand=quadro,driver>=560,driver<561 brand=quadrortx,driver>=560,driver<561 brand=nvidiartx,driver>=560,driver<561 brand=vapps,driver>=560,driver<561 brand=vpc,driver>=560,driver<561 brand=vcs,driver>=560,driver<561 brand=vws,driver>=560,driver<561 brand=cloudgaming,driver>=560,driver<561 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566
250325-10:54:29.877 INFO: env: NV_CUDA_CUDART_VERSION: 12.8.90-1
250325-10:54:29.877 INFO: env: CUDA_VERSION: 12.8.1
250325-10:54:29.877 INFO: env: NVIDIA_VISIBLE_DEVICES: GPU-af0684a1-a407-1553-0ec4-750609f59600
250325-10:54:29.877 INFO: env: NVIDIA_DRIVER_CAPABILITIES: compute,utility
250325-10:54:29.877 INFO: env: NV_CUDA_LIB_VERSION: 12.8.1-1
250325-10:54:29.877 INFO: env: NVIDIA_PRODUCT_NAME: CUDA
250325-10:54:29.878 INFO: listdir(/dev/nvidia-caps-imex-channels): ['channel0']
250325-10:54:29.878 INFO: /proc/devices contains IMEX devices: ['508 nvidia-caps-imex-channels']
250325-10:54:29.878 INFO: k8s job completion index: 1
250325-10:54:30.034 INFO: getDeviceCount(): 1
250325-10:54:30.037 INFO: device 0 properties:
{'major': 9,
 'minor': 0,
 'multiGpuBoardGroupID': 0,
 'multiProcessorCount': 132,
 'name': b'NVIDIA GH200 96GB HBM3',
 'pciBusID': 1,
 'pciDeviceID': 0,
 'pciDomainID': 9,
 'uuid': UUID('af0684a1-a407-1553-0ec4-750609f59600')}
250325-10:54:30.037 INFO: enter loop: get fabric handle from leader, URL: http://repro2-0.svc-repro2:1337/fabric-handle
250325-10:54:30.043 INFO: request failed, retry soon -- err was: HTTPConnectionPool(host='repro2-0.svc-repro2', port=1337): Max retries exceeded with url: /fabric-handle (Caused by NameResolutionError("<urllib3.connection.HTTPConnection object at 0xec92cbde3470>: Failed to resolve 'repro2-0.svc-repro2' ([Errno -2] Name or service not known)"))
250325-10:54:33.046 INFO: leader responded, response body (hex prefix): 4141416741...
250325-10:54:33.046 INFO: fabric handle data: b'\x00\x00 \x00\x00\x00\x00\x00\x82\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
250325-10:54:33.046 ERROR: main() crashed, traceback:
Traceback (most recent call last):
  File "/thing/fabric-handle-transfer-test.py", line 359, in <module>
    main()
  File "/thing/fabric-handle-transfer-test.py", line 98, in main
    follower_raw_cuda_mem_share_over_imex(fabric_handle_data)
  File "/thing/fabric-handle-transfer-test.py", line 217, in follower_raw_cuda_mem_share_over_imex
    handle_local = checkCudaErrors(
                   ^^^^^^^^^^^^^^^^
  File "/thing/fabric-handle-transfer-test.py", line 343, in checkCudaErrors
    raise RuntimeError(
RuntimeError: CUDA error code=101(b'CUDA_ERROR_INVALID_DEVICE')
250325-10:54:33.047 INFO: exit 0
```