#!/usr/bin/python3.6

def load_labels():
    """
    Returns a dict of:
        - key: id of the sample
        - value: list of labels
    """
    with open("../data/rcv1-v2.topics.qrels") as f:
        data = f.readlines()
    data = map(lambda x: x.split()[:-1], data)
    data_keyed = {}
    for label, sample in data:
        sample = int(sample)
        if sample not in data_keyed.keys():
            data_keyed[sample] = [label]
        else:
            data_keyed[sample].append(label)
    return data_keyed

def get_batch(batch_size=20):
    batch = {}
    counter = 0
    with open("../data/data") as f:
        for sample in f:
            counter += 1
            sample = sample.split()
            sample_labels = [ (int(label_id), float(confidence)) for combo in sample[1:] for label_id, confidence in [combo.split(":")] ]
            sample_id = int(sample[0])

            batch[sample_id] = dict(sample_labels)

            if counter % batch_size == 0:
                yield batch
                batch = {}
