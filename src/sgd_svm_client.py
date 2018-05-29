import data
import grpc
import json
import os
import random
import svm_function
import time

import sgd_svm_pb2
import sgd_svm_pb2_grpc

# Loading up the configuration
_NUM_FEATURES = 47236
_LEARNING_RATE = float(os.environ['LEARNING_RATE'])
_STOPPING_CRITERION = svm_function.get_stopping_criteria(_LEARNING_RATE)
_TRAIN_TO_VALID_RATIO = int(os.environ['TRAIN_TO_VALID_RATIO'])
_MINI_BATCH_SIZE = int(os.environ['MINI_BATCH_SIZE'])
_ASYNCHRONOUS = os.environ['ASYNCHRONOUS'] in ["True","yes","true","y"]
_SERVER_URL = "server-0.hog-service:50051"

class Client(object):
    """
    The client connecting to the server and asking for
    data, to then compute an update for the weights and
    send it back to the server
    """
    def __init__(self, channel=grpc.insecure_channel(_SERVER_URL)):
        """
        Initializes the client object with the server stub, and initial
        values (not connected, and no id given by the server)
        """
        self.stub = sgd_svm_pb2_grpc.SGDSVMStub(channel)
        self.connected = False
        self.train_set = iter([])
        self.training_dataset = {}
        self.valid_set = iter([])
        self.labels = {}
        self.weights = {str(feat):0 for feat in range(1, _NUM_FEATURES + 1)}
        self.last_grads_update = {}
        self.id = -1

    def getDataFromServer(self):
        """
        Asks data from the server, with our authentication id

        Returns:
            - samples as a string json
            - labels as a string json
            - weights as a string json
        """
        response_iterator = self.stub.getDataLabels(sgd_svm_pb2.Auth(id=self.id))
        self.training_dataset = {}
        while True:
            try:
                response = next(response_iterator)
                self.training_dataset.update(json.loads(response.samples_json))
                self.labels.update(json.loads(response.labels_json))
            except StopIteration:
                break

        # Split into training and validation sets
        if _TRAIN_TO_VALID_RATIO != 0:
            split_ = int(len(self.training_dataset)/_TRAIN_TO_VALID_RATIO)
            self.train_set = data.grouper(_MINI_BATCH_SIZE, list(self.training_dataset.items())[:split_])
            self.valid_set = data.grouper(_MINI_BATCH_SIZE*_TRAIN_TO_VALID_RATIO, list(self.training_dataset.items())[split_:])
        else:
            self.train_set = data.grouper(_MINI_BATCH_SIZE, list(self.training_dataset.items()))

        print("Client {}: All data downloaded and split in training and validation sets".format(self.id))

    def sendGradientUpdateToServer(self, grad_update):
        """
        Sends the computed updated gradient
        """
        updated_gradient = sgd_svm_pb2.GradientUpdate(gradient_update_json=json.dumps(grad_update), id=self.id)
        return self.stub.sendGradientUpdate(updated_gradient)

    def sendDoneComputingToServer(self):
        """
        Send a message to the server telling it that we are
        done computing for the specific task
        """
        return self.stub.sendDoneComputing(sgd_svm_pb2.Auth(id=self.id))

    def getGradsToServer(self):
        return self.stub.getGrads(sgd_svm_pb2.Auth(id=self.id))

    def authToServer(self):
        """
        Sends an authentication request to the server,
        if the server as stil some room for the client,
        it will assign it an id, otherwise, the id will be -1
        """
        auth = self.stub.auth(sgd_svm_pb2.Empty())
        if auth.id != -1:
            self.connected = True
            self.id = auth.id
            print("Client {} connected".format(self.id))
        else:
            print("ERROR, this client cannot connect to server")

    def shouldWaitSynchronousOrNotToServer(self):
        """
        Asks the server wether or not the client should wait
        before recomputing a new update during the synchronous
        phase
        """
        return self.stub.shouldWaitSynchronousOrNot(sgd_svm_pb2.Auth(id=self.id))

    def work(self):
        """
        While there are samples given by the server, this function
        will iterate over them to compute updates for the gradient,
        and send them back to the aggregator (server)
        """
        print("\n-------------- Connecting to the server --------------")
        # Authenticate
        self.authToServer()
        print("--------------      Connected       --------------")
        self.getDataFromServer()
        elapsed = time.time()
        if _TRAIN_TO_VALID_RATIO != 0:
            number_batches = int(len(self.training_dataset) / (1+_TRAIN_TO_VALID_RATIO) / _MINI_BATCH_SIZE)
        else:
            number_batches = int(len(self.training_dataset) / _MINI_BATCH_SIZE)

        epoch = 0
        previous_valid_loss = 1
        valid_loss = 0.9
        accumulated_train_losses = []
        accumulated_valid_losses = []
        while previous_valid_loss - valid_loss > _STOPPING_CRITERION:
            if epoch > 1:
                previous_valid_loss = valid_loss
            iteration = 1
            while True:
                # Verify that we are still authenticated
                try:
                    assert self.connected, "ERROR: client {} not connected to the server".format(self.id)
                except AssertionError:
                    break

                # load the data into variables and compute update
                # to send it back.
                try:
                    train_batch = dict(next(self.train_set))
                    if _TRAIN_TO_VALID_RATIO != 0:
                        valid_batch = dict(next(self.valid_set))
                except StopIteration:
                    break

                # separately compute train and valid losses
                train_loss = svm_function.calculate_loss(self.labels, train_batch, self.weights)
                if _TRAIN_TO_VALID_RATIO != 0:
                    valid_loss = svm_function.calculate_loss(self.labels, valid_batch, self.weights)
                else:
                    valid_loss = 0

                grad_update = svm_function.mini_batch_update(train_batch, self.labels, self.weights)

                # send back train gradient update
                self.sendGradientUpdateToServer(grad_update)

                if _ASYNCHRONOUS:
                    response = self.getGradsToServer()
                    received_grads = json.loads(response.grads)
                    if received_grads != self.last_grads_update:
                        self.last_grads_update = received_grads
                        for weight_id in received_grads:
                            self.weights[weight_id] -= _LEARNING_RATE*received_grads[weight_id]
                else:
                    while self.shouldWaitSynchronousOrNotToServer().answer:
                        time.sleep(0.0000000000001)
                    response = self.getGradsToServer()
                    received_grads = json.loads(response.grads)
                    self.last_grads_update = received_grads
                    for weight_id in received_grads:
                        self.weights[weight_id] -= _LEARNING_RATE*received_grads[weight_id]
                if iteration%10 == 0:
                    print('epoch {:2d} | {:5.2f}s elapsed | {:3d}/{:3d} batch | train_loss {:6.4f} | valid_loss {:6.4f}'.
                        format(epoch+1, time.time() - elapsed, iteration, number_batches, train_loss, valid_loss))

                iteration += 1

            # reset the training and validation sets after each epoch
            if _TRAIN_TO_VALID_RATIO != 0:
                split_ = int(len(self.training_dataset)/_TRAIN_TO_VALID_RATIO)
                self.train_set = data.grouper(_MINI_BATCH_SIZE, list(self.training_dataset.items())[:split_])
                self.valid_set = data.grouper(_MINI_BATCH_SIZE*_TRAIN_TO_VALID_RATIO, list(self.training_dataset.items())[split_:])
            else:
                self.train_set = data.grouper(_MINI_BATCH_SIZE, list(self.training_dataset.items()))

            accumulated_train_losses.append(train_loss)
            accumulated_valid_losses.append(valid_loss)


            epoch += 1
        self.sendDoneComputingToServer()
        print("training losses:", accumulated_train_losses)
        print("validation losses:", accumulated_valid_losses)

if __name__ == '__main__':
    # Run the client
    while True:
        try:
            client = Client()
            client.work()
        except:
            time.sleep(2)
            continue
        else:
            break
    print("\n-------------- Disconnected from the server --------------")
    # Sleep after finishing up experiments
    # to prevent a kubernetes container loop
    time.sleep(1000)
