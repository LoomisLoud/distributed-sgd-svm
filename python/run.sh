#!/bin/sh

# Choose your number of clients
NUM_CLIENTS=${@: -1}
echo "Starting the clients:"
for ((n=0;n<$NUM_CLIENTS;n++))
do
    python sgd_svm_client.py &
done
