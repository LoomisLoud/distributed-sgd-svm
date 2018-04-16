import grpc
import json
import random
import svm_function

import sgd_svm_pb2
import sgd_svm_pb2_grpc


class Client(object):
    """
    The client connecting to the server and asking for
    data, to then compute an update for the weights and
    send it back to the server
    """
    def __init__(self, channel=grpc.insecure_channel('localhost:50051')):
        """
        Initializes the client object with the server stub, and initial
        values (not connected, and no id given by the server)
        """
        self.stub = sgd_svm_pb2_grpc.SGDSVMStub(channel)
        self.connected = False
        self.id = -1

    def getDataFromServer(self):
        """
        Asks data from the server, with our authentication id

        Returns:
            - samples as a string json
            - labels as a string json
            - weights as a string json
        """
        return self.stub.getDataLabels(sgd_svm_pb2.Auth(id=self.id))

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

    def work(self):
        """
        While there are samples given by the server, this function
        will iterate over them to compute updates for the gradient,
        and send them back to the aggregator (server)
        """
        print("\n-------------- Connection to server --------------")
        # Authenticate
        self.authToServer()
        print("--------------      Connected       --------------")
        while True:
            # Verify that we are still authenticated
            try:
                assert self.connected, "ERROR: client {} not connected to the server".format(self.id)
            except AssertionError:
                break

            response = self.getDataFromServer()
            # If the server answers with empty data, disconnect
            if not response.samples_json:
                print("Client {} disconnecting from server".format(self.id))
                self.sendDoneComputingToServer()
                self.connected = False
                break

            # load the data into variables and compute update
            # to send it back.
            samples = json.loads(response.samples_json)
            labels = json.loads(response.labels_json)
            weights = json.loads(response.weights_json)

            loss = svm_function.calculate_loss(labels, samples, weights)
            print("Loss on clients: {:.6f}".format(loss), end="\r")
            grad_update = svm_function.mini_batch_update(samples, labels, weights)
            self.sendGradientUpdateToServer(grad_update)

if __name__ == '__main__':
    # Run the client
    client = Client()
    client.work()
