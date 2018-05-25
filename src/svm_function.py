from functools import reduce
#import matplotlib.pyplot as plt
"""
This helper file contains everything related to
SGD and SVM computations. Simply import the file
and use the functions with their defined arguments.
"""

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

def prepare_standardplot(title, xlabel, logscale):
    """
    Initialize plots
    """
    fig, ax1 = plt.subplots()
    fig.suptitle(title)
    ax1.set_ylabel('hinge loss')
    ax1.set_xlabel(xlabel)
    if logscale:
        ax1.set_yscale('log')
    return fig, ax1

def scriteria_to_lrate(stop_cri = 0.001, lag = 0):
    c = 0.5  
    L = 0.1
    M = 0.1
    omega = 50
    rho = 0.44
    delta = 1.
    phi = 0.9
    learning_rate = phi*stop_cri*c/(2*L*M*M*omega*(1 + 6*lag*rho + 4*lag*lag*omega*np.sqrt(delta)))
    return learning_rate 

def plot_history(train,test,title):
    """
    Plots learning curves
    train & test, list
    """
    fig, ax1 = prepare_standardplot(title, 'epoch', logscale=False)
    ax1.plot(train, 'b', label="training")
    ax1.plot(test, 'r', label="validation")
    ax1.legend()
    plt.savefig(str(title)+'.png')

def calculate_loss(labels, samples, weights):
    """
    Computes loss objective function of Support Vector Machine

    samples, dict{sample_id : dict{feat_id : val}}: containing the samples you want to use to compute the loss
    labels, dict{sample_id : label}: +1 or -1 labels. shape = (num_sample)
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
    return reduce(add_all, hinge_loss_by_sample)/len(samples)

def calculate_accuracy(labels, samples, weights):
    """
    Computes accuracy objective function of Support Vector Machine

    samples, dict{sample_id : dict{feat_id : val}}: containing the samples you want to use to compute the loss
    labels, dict{sample_id : label}: +1 or -1 labels. shape = (num_sample)
    weights, dict{feat_id : val}: shape = (num_features)
    """
    weighted_sum_samples = {}

    for key in samples.keys():
        feats = list(samples[key].keys())
        for i in feats:
            if i not in weights.keys():
                print(list(weights.keys())[:100])
                print(i)
                break
        sample_weight = [(samples[key][i], weights[i]) for i in feats]
        weighted_sum_samples[key] = reduce(add_all , map(multiply , sample_weight))

    sample_ids = list(weighted_sum_samples.keys())
    label_weighted_sum = [(labels[i], weighted_sum_samples[i]) for i in sample_ids]
    pred = map(lambda arg: (arg[0], +1) if arg[1] >= 0 else (arg[0], -1), label_weighted_sum)
    accuracy = reduce(add_all, map(lambda arg: 1 if arg[0] == arg[1] else 0 , pred)) / len(samples)
    return accuracy

def is_support(label, sample, weights):
    """
    Function that returns true if the sample is in the support of the hinge function

    Args:
        label, {-1,+1}: The label of the sample
        sample, dict: feature values of the sample.
        weights, dict : the weight vector.

    Returns:
        Bool: True when sample is in the support, False otherwise.
    """
    sample_weight = [(sample[i], weights[i]) for i in sample.keys()]
    dot_prod = reduce(add_all , map(multiply , sample_weight))
    return dot_prod*label < 1

def mini_batch_update(batch_, final_labels, weights, lam_ = 0.):

    """
    Function that returns the gradient update given multiple samples
    If the sample is not in the support, don't update the gradient (None) for this specific sample
    Args:
        batch_: dict{sample_id : dict{feat_id : val}}
        final labels: dict{sample_id : label}: +1 or -1 labels. shape = (num_sample)
        weights: dict{feat_id : val}: shape = (num_features)

    Returns:
        dict(feat_id: update): the gradient update
    """
    keys_ = list(batch_.keys())
    a = [(i, batch_[i], final_labels[i]) for i in keys_]
    b = list(map(lambda item: {k: -item[2]*v for k, v in item[1].items()} if is_support(item[2], item[1], weights) else None, a))
    filtered_b = [x for x in b if x is not None]
    if filtered_b:
        return reduce((lambda x, y: {**x, **y}), filtered_b)
    else:
        return { key:0 for key in weights.keys() }
    
def mini_batch_update_new(batch_, final_labels, weights, lam_ = 0.):

    """
    Function that returns the gradient update given multiple samples
    If the sample is not in the support, don't update the gradient (None) for this specific sample
    Args:
        batch_: dict{sample_id : dict{feat_id : val}}
        final labels: dict{sample_id : label}: +1 or -1 labels. shape = (num_sample)
        weights: dict{feat_id : val}: shape = (num_features)

    Returns:
        dict(feat_id: update): the gradient update
    """
    keys_ = list(batch_.keys())
    a = [(i, batch_[i], final_labels[i]) for i in keys_]
    b = list(map(lambda item: {k: -item[2]*v for k, v in item[1].items()} if is_support(item[2], item[1], weights) else None, a))
    filtered_b = [x for x in b if x is not None]
    if filtered_b:
        c = reduce((lambda x, y: {**x, **y}), filtered_b)
        d = [(c[i], weights[i]) for i in keys_]
        return reduce(lambda y: y[0]+y[1] , list(map(lambda y: y[0]+y[1],d)))
    else:
        return { key:0 for key in weights.keys() }
