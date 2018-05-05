#!/usr/bin/python3.6
"""
The aim of this helper module is simply to load the
data into the server. The labels file is quite small
and hence is loaded fully in memory, while the training
data is huge and while possible to load in memory, it
is quite costly. This is why the get_batch function
defines an iterator on which we can query for samples
"""

def load_labels():
    """
    Loads the labels from disk into a dict

    Returns a dict of:
        - key: id of the sample
        - value: list of labels
    """
    with open("../data/rcv1-v2.topics.qrels") as f:
        data = f.readlines()
    data = map(lambda x: x.split()[:-1], data)
    data_keyed = {}
    # Building a dict of labels
    for label, sample in data:
        if sample not in data_keyed.keys():
            data_keyed[sample] = [label]
        else:
            data_keyed[sample].append(label)
    return data_keyed

def get_batch(batch_size=1):
    """
    Loads the training data from disk into
    an iterator used by the server to serve
    batches of training data

    Returns:
        - iterator over the data
    """
    batch = {}
    counter = 0
    with open("../data/train_set") as f:
        for sample in f:
            # count the samples to send only a modulo
            # of the batch size asked by the user
            counter += 1
            sample = sample.split()
            sample_labels = [ (label_id, float(confidence)) for combo in sample[1:] for label_id, confidence in [combo.split(":")] ]
            sample_id = sample[0]
            batch[sample_id] = dict(sample_labels)
            if counter % batch_size == 0:
                yield batch
                batch = {}

def load_test_set():
    """
    Loads the testing data from disk into
    a dict used by the server to compute
    the accuracy

    Returns:
        - test set as a dict of samples
    """
    test_set = {}
    with open("../data/test_set") as f:
        for sample in f:
            sample = sample.split()
            sample_labels = [ (label_id, float(confidence)) for combo in sample[1:] for label_id, confidence in [combo.split(":")] ]
            sample_id = sample[0]
            test_set[sample_id] = dict(sample_labels)
    return test_set
