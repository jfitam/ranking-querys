from collections.abc import Sequence
from typing import Union
from numpy.typing import NDArray
import numpy as np
import numbers
from .ndcg_score import ndcg_score
import warnings

def ndcg_scores_set(scores: Sequence[numbers.Real] | NDArray[np.floating], 
                    y: Sequence[numbers.Real] | NDArray[np.floating], 
                    groups: Sequence[numbers.Real] | NDArray[numbers.Real],
                   verbose = -1) -> tuple[list[float], int]:
    '''Calculate the NDCG for a full test set, as the average of the NDCG of all the querys
    
    Args:
        scores (list): List of scores for each document evaluated
        y (list): The real relevancy for those documents
        groups (list): the amount of entries for each group, in a list of the same order than the data.
    
    Returns:
        A pairwise with the list of all the NDCG scores for all the querys of the input, and the number of querys processed
    '''
    #convert
    scores = np.array(scores)
    y = np.array(y)
    
    current_position = 0
    N = len(groups)
    results = []
    
    for k in groups:
        sc_sliced = scores[current_position:current_position+k]
        y_sliced = y[current_position:current_position+k]
        current_position += k
        if len(sc_sliced) != len(y_sliced):
            if verbose >= 0:
                warnings.warn("Error slicing the arrays for group processing. Returning the result so far.")
            return results, N

        if y_sliced.sum() < 1:
            if verbose >= 0:
                warnings.warn("Found a list of relevances with no values. Skipping.")
            N -= 1
            continue
            
        results.append( ndcg_score(sc_sliced, y_sliced))


    return results, N  
        