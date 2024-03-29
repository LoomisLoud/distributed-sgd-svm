from concurrent import futures
from math import floor
import data
import grpc
import json
import os
import svm_function
import sys
import threading
import time

import sgd_svm_pb2
import sgd_svm_pb2_grpc

_NUM_FEATURES = 47236
# Loading up the configuration
_NUM_TOTAL_CLIENTS = int(os.environ['CLIENTS'])
_LEARNING_RATE = float(os.environ['LEARNING_RATE'])


class SGDSVM(sgd_svm_pb2_grpc.SGDSVMServicer):
    """
    The gRPC server and aggregator. Its job is to manage the data
    on the master node, and send it to each worker for them to do the
    heavy computation
    """

    def __init__(self, nb_clients):
        """
        Initializes the server object with a few parameters:
            - total_clients: number of maximum clients handled by the server
            - connected_clients: number of currently connected clients
            - lock: simple threading lock used to lock the access to the iterator
                    to avoid concurrent access issues
            - data: the training data
            - labels: dict of labels for all of the samples
            - weights: current weight vector for SGD
            - grad_cumulator: cumulator used to cumulate the updates sent by each node
            - gradients_received: list of client ids who sent their gradients update
                                  for the current iteration of the algorithm
        """
        self.total_clients = nb_clients
        self.connected_clients = 0

        self.lock = threading.Lock()
        print("Loading the training set...")
        self.data = data.get_batch(floor(data.get_data_size() / nb_clients))
        self.labels = dict([ svm_function.contains_CCAT(tup) for tup in data.load_labels().items()])
        print("Loading the testing set...")
        def load_test():
            self.test_set = data.load_test_set()
        self.thread = threading.Thread(target=load_test, args=())
        self.thread.daemon = True
        self.thread.start()

        self.weights = {str(feat):0 for feat in range(1, _NUM_FEATURES + 1)}
        self.grad_cumulator = {}
        self.sendable_grads = {}
        self.gradients_received = []

        self.train_loss_received = []
        self.train_loss = 0
        self.valid_loss_received = []
        self.valid_loss = 0

        self.current_sample = 0

    def waitOnAllClientConnections(self):
        """
        This function forces the nodes to sleep at the start of the very
        first iteration, until all expected nodes connected to the server
        """
        while self.connected_clients < self.total_clients:
            continue

    def shouldWaitSynchronousOrNot(self, request, context):
        return sgd_svm_pb2.Answer(answer=request.id in self.gradients_received)

    def getDataLabels(self, request, context):
        """
        Checks if the client is eligible to retrieve a batch of data, and
        if so, returns the data.

        Returns a gRPC data object containing:
            - samples as a json
            - labels as a json
            - weights as a json
        """
        self.waitOnAllClientConnections()

        # Lock the iterator to prevent concurrent access
        print("Sending data and labels to client {}".format(request.id))
        try:
            with self.lock:
                samples = next(self.data)
                self.current_sample += len(samples)
                batch = {}
                for i, s in enumerate(samples):
                    batch.update({s:samples[s]})
                    if len(batch) == 1000 or i == len(samples) - 1:
                        labels = {key:self.labels[key] for key in batch.keys()}
                        yield sgd_svm_pb2.Data(samples_json=json.dumps(batch), labels_json=json.dumps(labels))
                        batch = {}
        except RuntimeError:
            pass

    def sendGradientUpdate(self, request, context):
        """
        Retrieves the updated gradients from all workers,
        sum them up. If we receive the last gradient update,
        update the total weights of the algorithm, reset the
        weights accumulator and the waiting list of workers
        """
        # register that this client computed
        self.gradients_received.append(request.id)
        grad_update = json.loads(request.gradient_update_json)
        # if the gradient update is not empty
        with self.lock:
            for weight_id in grad_update.keys():
                # accumulate gradients of all nodes
                if weight_id in self.grad_cumulator.keys():
                    self.grad_cumulator[weight_id] += grad_update[weight_id]
                else:
                    self.grad_cumulator[weight_id] = grad_update[weight_id]

            # if we are done with the current iteration,
            # update the weights and reset the cumulator of weights
            if len(self.gradients_received) >= self.connected_clients:
                for weight_id in self.grad_cumulator:
                    self.weights[weight_id] -= _LEARNING_RATE*self.grad_cumulator[weight_id]

                self.sendable_grads = self.grad_cumulator
                self.grad_cumulator = {}
                self.gradients_received = []

        return sgd_svm_pb2.Empty()

    def sendDoneComputing(self, request, context):
        """
        Tells the server that the worker did not get any
        data from the iterator, and that it is done computing
        """
        print("Client {} is done computing".format(request.id))
        self.connected_clients -= 1
        self.gradients_received.append(request.id)
        # if we are the last client, compute
        # the accuracy and display it
        if self.connected_clients == 0:
            print("Computing accuracy...")
            # Computing accuracy
            self.thread.join()
            split = int(len(self.test_set)/_NUM_TOTAL_CLIENTS)
            labels = {key:self.labels[key] for key in self.test_set.keys()}
            accuracy = svm_function.calculate_accuracy(labels, self.test_set, self.weights)

            # Printing the training and testing loss
            print("Computed accuracy: {:.2f}%".format(accuracy*100))

        return sgd_svm_pb2.Empty()

    def getGrads(self, request, context):
        """
        Retrieves the weights from the server
        """
        return sgd_svm_pb2.Grads(grads=json.dumps(self.sendable_grads))

    def auth(self, request, context):
        """
        Authenticates to the server, if the number of
        connected clients is less that the total number
        of expected clients, we can allow the client to
        authenticate. Otherwise, send an error and a bad id

        Returns a gRPC Auth object containing:
            - id: id of the authenticated client
        """
        if self.connected_clients < self.total_clients:
            self.connected_clients += 1
            print("Client {} connected".format(self.connected_clients))
            return sgd_svm_pb2.Auth(id=self.connected_clients)
        else:
            print("ERROR, can't auth new client, maximum threshold of clients reached.")
            return sgd_svm_pb2.Auth(id=-1)

def serve(clients):
    """
    Serves the gRPC messaging server, and defines
    how many clients we allow to connect to the server
    """
    print("Starting the server ...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=clients))
    sgd_svm_pb2_grpc.add_SGDSVMServicer_to_server(
        SGDSVM(nb_clients=clients), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started")
    # Sleep after finishing up experiments
    # to prevent a kubernetes container loop
    time.sleep(1000)
    server.stop(0)
    print("Server stopped")

if __name__ == '__main__':
    serve(clients=_NUM_TOTAL_CLIENTS)
