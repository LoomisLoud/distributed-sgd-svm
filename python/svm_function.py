import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import data

dict2list = lambda dic: [(k, v) for (k, v) in dic.items()]
list2dict = lambda lis: dict(lis)
multiply = lambda item: (item[0]*item[1])
add_all = lambda y, z: (y + z)

def contains_CCAT(item):
    if 'CCAT' in item[1]:
        return (item[0], 1)
    else:
        return (item[0], -1)
    

def calculate_loss(y, X, w, lambda_=0):
    """
    Compute loss objective function of Support Vector Machine
    
    X (dict{dict}): containing the samples you want to use to compute the loss
    y (dict): +1 or -1 labels of samples, but can be set as (list) before the function, shape = (num_sample)
    w (dict): shape = (num_features)
    lambda_ (double): In case we add regularization
    """
    r={}
    
    for key in X.keys():
        feats = list(X[key].keys())
        l = [(X[key][i], w[i]) for i in feats]
        
        dot_prod = reduce(add_all , map(multiply , l))
        r[key] = (dot_prod)
    
    samples = list(r.keys())
    a = [(y[i], r[i]) for i in samples]
    hinge = lambda arg: max(0, 1-arg[0]*arg[1])
    sum_ = reduce(add_all, map(hinge , a))
    
    return sum_

def is_support(y_n, x_n, w):
    """Function that true if the sample is in the support of the hinge function

    Args:
        y_n ({-1,+1}): The label of the sample
        x_n (dict): label values of the sample.
        w (dict) : the weight vector.

    Returns:
        Bool: The return True when sample is in the support, False otherwise.
    """
    feats = list(x_n.keys())
    l = [(x_n[i], w[i]) for i in feats]
    multiply = lambda item: (item[0]*item[1])
    add_all = lambda x, y: (x + y)
    dot_prod = reduce(add_all , map(multiply , l))
    return dot_prod*y_n < 1 

def gradient_update(y_n, x_n, w):
    """Function that return the gradient update 
    If the sample is not in the support, don't update the gradient (None)
    
    Returns:
        dict: The gradient update with (key,value)=(label_id, update)
    """
    if(is_support(y_n, x_n, w)):
        grad_update = list2dict(map(lambda item: (item[0],-y_n * item[1]) , dict2list(x_n)))
    else:
        grad_update = None
            
    return (grad_update)

