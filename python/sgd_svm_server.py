from concurrent import futures
import json
import threading
import time

import grpc

import sgd_svm_pb2
import sgd_svm_pb2_grpc
import svm_function
import data

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_NUM_TOTAL_CLIENTS = 2
_NUM_FEATURES = 47236
_NUM_SAMPLES = 781265
_LEARNING_RATE = 0.05

class SGDSVM(sgd_svm_pb2_grpc.SGDSVMServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self, nb_clients):
        self.total_clients = nb_clients
        self.connected_clients = 0

        self.lock = threading.Lock()
        self.data = iter(data.get_batch(1))
        self.labels = dict([ svm_function.contains_CCAT(tup) for tup in data.load_labels().items()])

        self.weights = {str(feat):0 for feat in range(_NUM_FEATURES)}
        self.weights_cumulator = {str(feat):0 for feat in range(_NUM_FEATURES)}
        self.gradients_received = []

    def waitOnAllClientConnections(self):
        while self.connected_clients < self.total_clients:
            time.sleep(1)

    def waitOnGradientUpdates(self, id):
        while id in self.gradients_received:
            continue

    def getDataLabels(self, request, context):
        self.waitOnAllClientConnections()
        self.waitOnGradientUpdates(request.id)
        try:
            with self.lock:
                samples = next(self.data)
        except StopIteration:
            return sgd_svm_pb2.Empty()
        labels = {key:self.labels[key] for key in samples.keys()}
        return sgd_svm_pb2.Data(samples_json=json.dumps(samples), labels_json=json.dumps(labels), weights_json=json.dumps(self.weights))

    def sendGradientUpdate(self, request, context):
        self.gradients_received.append(request.id)
        grad_update = json.loads(request.gradient_update_json)
        if grad_update:
            for weight_id in grad_update.keys():
                self.weights_cumulator[weight_id] += grad_update[weight_id]
                if len(self.gradients_received) == self.total_clients:
                    self.weights[weight_id] -= _LEARNING_RATE*self.weights_cumulator[weight_id]
        if len(self.gradients_received) == self.total_clients:
            self.weights_cumulator = {str(feat):0 for feat in range(_NUM_FEATURES)}
            self.gradients_received = []
        return sgd_svm_pb2.Empty()

    def sendDoneComputing(self, request, context):
        time.sleep(1)
        print("Done computing")
        self.connected_clients -= 1
        if self.connected_clients == 0:
            print("Computing loss")
            # Print the loss/accuracy ?
        return sgd_svm_pb2.Empty()

    def auth(self, request, context):
        if self.connected_clients < self.total_clients:
            self.connected_clients += 1
            print("Client {} connected".format(self.connected_clients))
            return sgd_svm_pb2.Auth(id=self.connected_clients)
        else:
            print("ERROR, too many clients connected".format(self.connected_clients))
            return sgd_svm_pb2.Auth(id=-1)

def serve(clients):
    print("Starting the server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=clients))
    sgd_svm_pb2_grpc.add_SGDSVMServicer_to_server(
        SGDSVM(nb_clients=clients), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        print("Server stopped")

if __name__ == '__main__':
    serve(clients=_NUM_TOTAL_CLIENTS)
