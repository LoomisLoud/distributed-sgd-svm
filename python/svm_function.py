from functools import reduce

multiply = lambda item: item[0]*item[1]
add_all = lambda y, z: y + z

def sub_dict(somedict, somekeys, default=None):
    return dict([ (k, somedict.get(k, default)) for k in somekeys ])

def contains_CCAT(item):
    """
    Returns true if the labels contain CCAT, false otherwise
    """
    if 'CCAT' in item[1]:
        return (item[0], 1)
    else:
        return (item[0], -1)

# To test this:
"""
labels = dict([ contains_CCAT(tup) for tup in data.load_labels().items()])
weights = [ random.random() for _ in range(48000) ]
samples = next(data.get_batch(20))

print(calculate_loss(labels, samples, weights))
print(gradient_update(labels[26151],samples[26151], weights))
print(is_support(labels[26151],samples[26151], weights))
"""
def calculate_loss(labels, samples, weights):
    """
    Compute loss objective function of Support Vector Machine

    samples, dict{sample_id : dict{feat_id : val}}: containing the samples you want to use to compute the loss
    final labels, dict{sample_id : label}: +1 or -1 labels. shape = (num_sample)
    weights, dict{feat_id : val}: shape = (num_features)
    """
    weighted_sum_samples = {}

    for key in samples.keys():
        feats = list(samples[key].keys())
        sample_weight = [(samples[key][i], weights[i]) for i in feats]
        weighted_sum_samples[key] = reduce(add_all , map(multiply , sample_weight))

    sample_ids = list(weighted_sum_samples.keys())
    label_weighted_sum = [(labels[i], weighted_sum_samples[i]) for i in sample_ids]
    # computing the hinge loss for each sample and summing
    hinge_loss_by_sample = map(lambda arg: max(0, 1-arg[0]*arg[1]), label_weighted_sum)
    return reduce(add_all, hinge_loss_by_sample)

def calculate_accuracy(labels, sample, weights):
    
    weighted_sum_samples = {}

    for key in samples.keys():
        feats = list(samples[key].keys())
        sample_weight = [(samples[key][i], weights[i]) for i in feats]
        weighted_sum_samples[key] = reduce(add_all , map(multiply , sample_weight))
        
    sample_ids = list(weighted_sum_samples.keys())
    label_weighted_sum = [(labels[i], weighted_sum_samples[i]) for i in sample_ids]
    pred = map(lambda arg: (arg[0], +1) if arg[1] >= 0 else (arg[0], -1), label_weighted_sum)
    accuracy = reduce(add_all, map(lambda arg: 1 if arg[0] == arg[1] else 0 , pred)) / len(samples)
    
    return accruacy

def is_support(label, sample, weights):
    """Function that true if the sample is in the support of the hinge function

    Args:
        label, {-1,+1}: The label of the sample
        sample, dict: feature values of the sample.
        weights, dict : the weight vector.

    Returns:
        Bool: The return True when sample is in the support, False otherwise.
    """
    sample_weight = [(sample[i], weights[i]) for i in sample.keys()]
    dot_prod = reduce(add_all , map(multiply , sample_weight))
    return dot_prod*label < 1

def gradient_update(label, sample, weights):
    """Function that return the gradient update
    If the sample is not in the support, don't update the gradient (None)
    
    Returns:
        dict: The gradient update with (key,value)=(label_id, update)
    """
    if is_support(label, sample, weights):
        grad_update = dict(map(lambda item: (item[0],-label * item[1]) , sample.items()))
    else:
        grad_update = None
    return grad_update

def mini_batch_update(batch_, final_labels, weights,mini_batch_size):
    
    """Function that return the gradient update given multiple samples
    If the sample is not in the support, don't update the gradient (None) for this specific sample

    Args:
        batch_, dict{sample_id : dict{feat_id : val}}
        final labels, dict{sample_id : label}: +1 or -1 labels. shape = (num_sample)
        weights, dict{feat_id : val}: shape = (num_features)
        
    Returns:
        dict(feat_id: update): the gradient update 
    """
    
    keys_ = list(batch_.keys())
    multiple_batch = sub_dict(batch_, keys_[:mini_batch_size])
    a = [(i, multiple_batch[i], final_labels[i]) for i in keys_[:mini_batch_size]]
    b = list(map(lambda item: {k: -item[2]*v for k, v in item[1].items()} if is_support(item[2], item[1], weights) else None, a))
    filtered_b = [x for x in b if x is not None]
    total_grad_update = reduce((lambda x, y: {**x, **y}), filtered_b)
    return total_grad_update