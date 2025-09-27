from collections.abc import Sequence
from typing import Union
from numpy.typing import NDArray
import numpy as np
import numbers
import warnings

def dcg_score(
    scores: Sequence[numbers.Real] | NDArray[np.floating], 
    y: Sequence[numbers.Real] | NDArray[np.floating],
        verbose = -1) -> float:
    
    '''Calculate the DCG metric for a result query prediction
    
    Args:
        scores (list): List of scores for each document evaluated
        y (list): The real relevancy for those documents
    
    Returns:
        A scalar equals to the calculated dcg metric
    '''
    
    #convert
    scores = np.array(scores, dtype=float)
    y = np.array(y, dtype=float)
    
    # get the n from y. adjust scores to the same size
    n = y.shape[0]

    if n < 1:
        warnings.warn("Shape of y below 1") 
        return None

    if scores.shape[0] != n:
        if verbose >= 0:
            warnings.warn("The shape of the ranking does not match the shape of the relevances. Adjusting it.")
        scores_padded = np.zeros(n)
        scores_padded[:len(scores)] = scores[:n]
        scores = scores_padded

    
    # normalize the relevances
    y = y + y.min()
    max_relevance = y.max()
    
    # relevances predicted 
    order = np.argsort(scores)
    predicted_relevances = y[order][::-1]

    if predicted_relevances.shape[0] < 1:
        warnings.warn("Error while predicting the order of the documents.")   
        
    #calculation
    dcg_value = 0
    for i in range(n):
        num = 2**predicted_relevances[i] - 1
        den = np.log2((i+1) + 1)
        dcg_value += num / den

    return dcg_value