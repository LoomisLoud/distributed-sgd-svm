# Distributed Support Vector Machine trained with Stochastic Gradient Descent
[Brune Bastide](mailto:brune.bastide@epfl.ch) (223967), [Augustin Prado](mailto:augustin.prado@epfl.ch) (237289), [Romain Choukroun](mailto:romain.choukroun@epfl.ch) (203917)

The aim of the project is to run a Support Vector Machine trained using Stochastic Gradient Descent. The current version of the project is being run synchronously, which we are going to explain later on.

## How to run the project

Store the training data in a file called `data` inside the data folder, alongside your testing set called `test_set` and the labels called `rcv1-v2.topics.qrels`. Install the required python libraries using `pip install -r requirements.txt` and setup gRPC by running: `python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sgd_svm.proto`.

To have a full view of what happens on the server and the clients, use two terminals, in one, run: `python sgd_svm_server.py X` replacing X with the number of clients you want and in the other, run the shell script which will launch all the clients and start the computation: `chmod +x run.sh && ./run.sh X`, using the same X as number of clients.

Running the code on 20 000 samples in the training set with 5 nodes and a mini batch of size 10 takes less than 2 minutes and returns an accuracy of 95%. Running on the full dataset with as small a number of workers as 5 should a bit less than an hour.

## Explanations and choices
### Machine learning wise
We have first labelled the dataset by looking whether each sample contains the word "CCAT" or not, assigning -1 and +1. The Support Vector Machine (SVM) algorithm tries to find the best hyperplane that separates our data points by maximising the margin -that is the distance between the hyperplane itself to hyperplane 's closest data point.

We have implemented in Python a stochastic gradient descent for this binary classification problem. We have used the standard SVM loss function: hinge loss function. Loss is computed using `calculate_loss` in `svm_function.py` file. Gradient is computed in function `mini_batch_update` where you can send samples one by one to do the formal stochastic gradient descent.

### Systems decisions
For this first milestone, we split the work among multiple workers. The whole data and weight vector (that uniquely determine a hyperplane) are both stored in the main server. This choice is justified since we are processing on one computer and every one has access to the same ram. These choices will obviously evolve as we move onto a cluster for the next milestone.

The server starts by waiting for a given number of client connections. As long as all clients are not connected, the server holds. When all clients have successfully connected, the server blocks further connections and at each iteration each worker receives a batch of data, alongside its corresponding labels and the current weight vector. Then, each worker node computes the gradient of the sample(s) it has just received using `mini_batch_update`  and sends the result back to the server. The server waits for each worker's answer and then update the weight vector by summing all the gradients received. Once the weight vector is updated, the iteration can start again.

When the work is done, the clients deauthenticate themselves from the server and the server computes the accuracy on the testing set using the weights.

## Future work and second milestone
There is still much to do, first maybe separating the server and the master node responsible for the resources allocation. We also need to randomize the iterator better, at the moment the data is randomized and split into training and testing sets through a shell script. An obvious other way to meliorate the project would be to add docker support.

Also we will still need to choose a different architecture for the next milestone.
