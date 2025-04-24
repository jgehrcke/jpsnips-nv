
```
$ docker buildx build \
    --progress plain . -t jgehrcke/nickelpie -f nickelpie.Dockerfile  && \
    docker push jgehrcke/nickelpie

...

#35 DONE 0.0s
Using default tag: latest
The push refers to repository [docker.io/jgehrcke/nickelpie]
```

```
$ docker image ls | grep nickel | head -n 5
jgehrcke/nickelpie                  latest                             94f4290a7a6f   12 minutes ago   1.07GB
```
