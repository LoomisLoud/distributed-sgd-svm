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
"""Common resources used in the gRPC route guide example."""

#import sgd_svm_pb2

def import_data():
    abalone_data = []
    with open("../data/abalone.data") as f:
        for item in f.readlines():
            token_list = [ float(i)  for i in item.split(",") ]
            abalone_data.append(token_list)
    return abalone_data
            #feature = route_guide_pb2.Feature(
            #    name=item["name"],
            #    location=route_guide_pb2.Point(
            #        latitude=item["location"]["latitude"],
            #        longitude=item["location"]["longitude"]))
            #feature_list.append(feature)
    #return feature_list

print(import_data())

