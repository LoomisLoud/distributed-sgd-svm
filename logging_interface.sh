#!/usr/bin/sh
. ./config_file;
$TERMINAL_USED -e bash -c "watch -n1 'kubectl get pods,statefulsets | tail -n 30'" &
$TERMINAL_USED -e bash -c "watch -n1 'kubectl logs --since=10h client-0 grpc-client | tail -n 30'" &
$TERMINAL_USED -e bash -c "watch -n1 'kubectl logs --since=10h client-1 grpc-client | tail -n 30'" &
watch -n1 'kubectl logs --since=10h server-0 grpc-server | tail -n 30'
