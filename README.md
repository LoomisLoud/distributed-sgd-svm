# Distributed Support Vector Machine trained with Stochastic Gradient Descent using Kubernetes (HOGWILD!)
[Brune Bastide](mailto:brune.bastide@epfl.ch), [Augustin Prado](mailto:augustin.prado@epfl.ch), [Romain Choukroun](mailto:romain.choukroun@gmail.com)

This is an example of  how to run a Support Vector Machine trained using Stochastic Gradient Descent distributed using Kubernetes/Docker. The current version of the project can be run in synchronous or asynchronous modes, which are explained in the report, see the report folder to learn more !

## How to run the project

The project exclusively runs in Kubernetes. You will need to create a persistent volume and change the claim name in the Kubernetes config file to fit it.

To have a full view of what happens in Kubernetes, on the server and two of the clients clients, change in the config file the *TERMINAL* variable by the name of your terminal, and run logging interface: `./logging_interface.sh`. You can setup your experiment in the config file, and when you're done, simply run: `./run_kub.sh` which will run the project in Kubernetes. You can follow the progress in the logging interface. If you have not uncommented the automatic termination in the `run_kub.sh` file, you can simply run: `./del_kub.sh` to terminate the experiment gracefully.

Running the code on 20 000 samples in the training set with 5 clients and a mini batch of size 16 takes less than a minute and returns an accuracy of 95%.

## Learn more about the project

To learn more about the project, do not hesitate to checkout our report in the report folder !
