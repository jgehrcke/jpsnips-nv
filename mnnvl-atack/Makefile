REV := $(shell git rev-parse --short=7 HEAD)
IMAGE := docker.io/jgehrcke/atack:$(REV)

.PHONY: build build-and-push build-and-push-as-latest follow-results scale-up scale-down clean

build:
	docker buildx build --progress plain -t $(IMAGE) .

build-and-push:
	docker buildx build --progress plain -t $(IMAGE) --push .

build-and-push-as-latest: build-and-push
	docker buildx build --progress plain -t docker.io/jgehrcke/atack:latest --push .

follow-results:
	python3 follow-results.py

scale-up:
	$(eval CURRENT := $(shell kubectl get statefulset atack -o jsonpath='{.spec.replicas}'))
	kubectl scale statefulset atack --replicas=$$(($(CURRENT) + 1))

scale-down:
	$(eval CURRENT := $(shell kubectl get statefulset atack -o jsonpath='{.spec.replicas}'))
	kubectl scale statefulset atack --replicas=$$(($(CURRENT) - 1))

clean:
	kubectl delete statefulset atack --ignore-not-found
	kubectl delete service svc-atack --ignore-not-found
	kubectl delete computedomain atack-compute-domain --ignore-not-found
