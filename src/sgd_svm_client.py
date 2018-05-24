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
_TEST_TO_TRAIN_RATIO = int(os.environ['TEST_TO_TRAIN_RATIO'])
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
        self.test_set = iter([])
        self.labels = {}
        self.weights = {str(feat):0 for feat in range(1, _NUM_FEATURES + 1)}
        self.last_weights_update = {}
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
        dataset = {}
        while True:
            try:
                response = next(response_iterator)
                dataset.update(json.loads(response.samples_json))
                self.labels.update(json.loads(response.labels_json))
            except StopIteration:
                break

        split_ = int(len(dataset)/_TEST_TO_TRAIN_RATIO)
        self.train_set = data.grouper(_MINI_BATCH_SIZE, list(dataset.items())[:split_])
        self.test_set = data.grouper(_MINI_BATCH_SIZE*_TEST_TO_TRAIN_RATIO, list(dataset.items())[split_:])

        print("Client {}: All data downloaded and split in training and testing sets".format(self.id))

    def sendGradientUpdateToServer(self, grad_update):
        """
        Sends the computed updated gradient
        """
        updated_gradient = sgd_svm_pb2.GradientUpdate(gradient_update_json=json.dumps(grad_update), id=self.id)
        return self.stub.sendGradientUpdate(updated_gradient)

    def sendEvalUpdateToServer(self, train_loss_update, test_loss_update):
        """
            Sends the computed updated gradient
            """
        updated_eval = sgd_svm_pb2.EvalUpdate(train_loss_update=train_loss_update, test_loss_update=test_loss_update, id=self.id)
        return self.stub.sendEvalUpdate(updated_eval)

    def sendDoneComputingToServer(self):
        """
        Send a message to the server telling it that we are
        done computing for the specific task
        """
        return self.stub.sendDoneComputing(sgd_svm_pb2.Auth(id=self.id))

    def getWeightsToServer(self):
        return self.stub.getWeights(sgd_svm_pb2.Auth(id=self.id))

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
            print("ERROR, client {} can't connect to server".format(self.id))

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
                test_batch = dict(next(self.test_set))
            except StopIteration:
                break

            # separately compute train and test losses
            train_loss = svm_function.calculate_loss(self.labels, train_batch, self.weights)
            test_loss = svm_function.calculate_loss(self.labels, test_batch, self.weights)

            print("train loss on client: {:.6f}".format(train_loss))
            grad_update = svm_function.mini_batch_update(train_batch, self.labels, self.weights)

            # send back train/test losses
            self.sendEvalUpdateToServer(train_loss, test_loss)

            # send back train gardient update
            self.sendGradientUpdateToServer(grad_update)

            if _ASYNCHRONOUS:
                response = self.getWeightsToServer()
                received_weights = json.loads(response.weights)
                if received_weights != self.last_weights_update:
                    self.last_weights_update = received_weights
                    for weight_id in received_weights:
                        self.weights[weight_id] -= _LEARNING_RATE*received_weights[weight_id]
            else:
                while self.shouldWaitSynchronousOrNotToServer().answer:
                    time.sleep(0.0000000000001)
                response = self.getWeightsToServer()
                received_weights = json.loads(response.weights)
                self.last_weights_update = received_weights
                for weight_id in received_weights:
                    self.weights[weight_id] -= _LEARNING_RATE*received_weights[weight_id]

        self.sendDoneComputingToServer()

if __name__ == '__main__':
    # Run the client
    while True:
        try:
            client = Client()
            client.work()
        except:
            time.sleep(1)
            continue
        else:
            break
    time.sleep(1000)
