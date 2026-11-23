import numpy as np
def soft_vote(probability_matrices, classes):
    if not probability_matrices:
        raise ValueError("probability_matrices cannot be empty")
    mean_prob = np.mean(np.stack(probability_matrices), axis=0)
    return np.asarray(classes)[mean_prob.argmax(axis=1)], mean_prob
