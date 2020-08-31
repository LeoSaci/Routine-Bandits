import numpy as np

def build_dataset(bandits, T, H, seq):
    """
    Return a python dictionary
    Each key period h gives a numpy array with shape (T, K), 
    where each column k is a list rewards 
    from the arm k of bandit b_h
    """
    K = bandits[0].K
    dataset = {}
    for h in range(H):
        samples_h = bandits[seq[h]].sample(T)
        dataset[h] = samples_h
    return dataset
