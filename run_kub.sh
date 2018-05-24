#!/usr/bin/sh
# loading the config
. ./config_file;
# replacing the variables in Kubernetes config
cat Kubernetes/pod-server.yaml |
sed 's/\$CLIENTS'"/$CLIENTS/g" |
sed 's/\$LEARNING_RATE'"/$LEARNING_RATE/g" |
sed 's/\$TEST_TO_TRAIN_RATIO'"/$TEST_TO_TRAIN_RATIO/g" |
sed 's/\$MINI_BATCH_SIZE'"/$MINI_BATCH_SIZE/g" |
sed 's/\$ASYNCHRONOUS'"/$ASYNCHRONOUS/g" |
# Running the experiment
kubectl create -f -;

# Opening the logging interface
#$TERMINAL_USED -hold -e bash -c "./logging_interface.sh" &

# Wait until the experiments are done
sleep 120;

# Delete the pods and everything else
./del_kub.sh;
