# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the gRPC route guide server."""

from concurrent import futures
import time

import grpc

import sgd_svm_pb2
import sgd_svm_pb2_grpc
import sgd_svm_resources

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class SGDSVM(sgd_svm_pb2_grpc.SGDSVMServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self):
        self.gradient_sum = 0
        self.gradients_received = 0
        self.total_clients = 0
        self.data = iter(sgd_svm_resources.import_data())

    def GetData(self, request, context):
        return sgd_svm_pb2.Data(chunk=next(self.data))

    def VerifyAddition(self, request, context):
        return sgd_svm_pb2.Data(chunk=request.chunk)


    #def AddGradient(self, gradient, context):
    #    if gradient is not None:
    #        self.gradient_sum += gradient
    #        self.gradients_received += 0
    #        return self.total_clients == self.gradients_received
    #    return None


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sgd_svm_pb2_grpc.add_SGDSVMServicer_to_server(
        SGDSVM(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
