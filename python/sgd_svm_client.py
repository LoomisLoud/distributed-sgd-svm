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
"""The Python implementation of the gRPC route guide client."""

import grpc

import sgd_svm_pb2
import sgd_svm_pb2_grpc
import sgd_svm_resources

#def get_update_gradient(stub, my_gradient):
#    done = stub.AddGradient(my_gradient)

def get_data_from_server(stub):
    return stub.GetDataLabels(sgd_svm_pb2.Empty())

def send_update_to_server(stub, gradients):
    return stub.VerifyAddition(sgd_svm_pb2.Data(chunk=gradients))

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = sgd_svm_pb2_grpc.SGDSVMStub(channel)
    print("-------------- GetData --------------")
    response = get_data_from_server(stub)
    print("Server answered:")
    print(response.chunk)
    tryout = [sum(response.chunk), 0, 0]
    response = send_update_to_server(stub, tryout)
    print("-------------- AddSend --------------")
    print("Server answered the sum of gradients:")
    print(response.chunk)

if __name__ == '__main__':
    run()
