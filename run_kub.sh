#!/usr/bin/sh

# Loading the config file
. ./config_file;

# Replacing the variables in the Kubernetes config
cat Kubernetes/pod-server.yaml |
sed 's/\$CLIENTS'"/$CLIENTS/g" |
sed 's/\$LEARNING_RATE'"/$LEARNING_RATE/g" |
sed 's/\$TRAIN_TO_VALID_RATIO'"/$TRAIN_TO_VALID_RATIO/g" |
sed 's/\$MINI_BATCH_SIZE'"/$MINI_BATCH_SIZE/g" |
sed 's/\$ASYNCHRONOUS'"/$ASYNCHRONOUS/g" |
# Running the experiment
kubectl create -f -;

# UNCOMMENT these lines if you want to
# automatically terminate the experience
# after it is done computing
# Wait until the experiments are done
#sleep 180;

# Delete the pods and everything else
#./del_kub.sh;
