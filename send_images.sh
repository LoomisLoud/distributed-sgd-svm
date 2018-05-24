#!/usr/bin/sh
docker ps -a | awk '{ print $1,$2 }' | grep grpc-client:latest | awk '{print $1 }' | xargs -I {} docker rm -f {}
docker ps -a | awk '{ print $1,$2 }' | grep grpc-server:latest | awk '{print $1 }' | xargs -I {} docker rm -f {}
docker rmi grpc-server grpc-client
docker rmi $(docker images -f "dangling=true" -q)

docker build -t grpc-server -f Docker/Dockerfile-server .
docker build -t grpc-client -f Docker/Dockerfile-client .

docker tag grpc-client:latest ic-registry.epfl.ch/systems/grpc-client:latest
docker tag grpc-server:latest ic-registry.epfl.ch/systems/grpc-server:latest

docker push ic-registry.epfl.ch/systems/grpc-client:latest
docker push ic-registry.epfl.ch/systems/grpc-server:latest
