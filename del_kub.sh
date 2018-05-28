#!/usr/bin/sh
# Gracefully terminates the experiment
kubectl delete --all pods,services,statefulsets
