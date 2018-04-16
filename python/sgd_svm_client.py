import grpc
import json
import random
import svm_function

import sgd_svm_pb2
import sgd_svm_pb2_grpc

class Client(object):
    def __init__(self, channel=grpc.insecure_channel('localhost:50051')):
        self.stub = sgd_svm_pb2_grpc.SGDSVMStub(channel)
        self.connected = False
        self.id = -1

    def getDataFromServer(self):
        return self.stub.getDataLabels(sgd_svm_pb2.Auth(id=self.id))

    def sendGradientUpdateToServer(self, grad_update):
        updated_gradient = sgd_svm_pb2.GradientUpdate(gradient_update_json=json.dumps(grad_update), id=self.id)
        return self.stub.sendGradientUpdate(updated_gradient)

    def sendDoneComputingToServer(self):
        return self.stub.sendDoneComputing(sgd_svm_pb2.Empty())

    def authToServer(self):
        auth = self.stub.auth(sgd_svm_pb2.Empty())
        if auth.id != -1:
            self.connected = True
            self.id = auth.id
            print("Client", self.id, "connected")
        else:
            print("ERROR, client", self.id, "can't connect to server")

    def work(self):
        print("-------------- Connection to server --------------")
        self.authToServer()
        while True:
            try:
                assert self.connected, "ERROR: client {} not connected to the server".format(self.id)
            except AssertionError:
                break
            print("-------------- GetData --------------")
            response = self.getDataFromServer()
            if not response.samples_json:
                print("Client", self.id, "disconnecting from server")
                self.sendDoneComputingToServer()
                self.connected = False
                break
            samples = json.loads(response.samples_json)
            labels = json.loads(response.labels_json)
            weights = json.loads(response.weights_json)
            current_key = list(samples.keys())[0]
            print("Loss on client", self.id, ":", svm_function.calculate_loss(labels, samples, weights))

            grad_update = svm_function.gradient_update(labels[current_key],samples[current_key], weights)
            print("-------------- SendUpdate --------------")
            self.sendGradientUpdateToServer(grad_update)

if __name__ == '__main__':
    client = Client()
    client.work()
