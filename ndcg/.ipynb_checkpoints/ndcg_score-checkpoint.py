from collections.abc import Sequence
from typing import Union
from numpy.typing import NDArray
import numpy as np
import numbers
from .dcg_score import dcg_score
import warnings

def ndcg_score(
    scores: Sequence[numbers.Real] | NDArray[np.floating], 
    y: Sequence[numbers.Real] | NDArray[np.floating],
        ) -> float:
    '''Calculate the Normalized DCG

    Args:
        scores (list): List of scores for each document evaluated
        y (list): The real relevancy for those documents
    
    Returns:
        A scalar equals to the calculated ndcg metric
    '''
    num = dcg_score(scores, y)
    den = dcg_score(y, y)

    if den == 0:
        warnings.warn("Real relevance equal zero. Returning full score.")
        return 1
        
    return  num / den