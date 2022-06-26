import numpy as np

def reinforce_score_function(lprob: np.ndarray,
                             ret: np.ndarray):
    return lprob * ret
