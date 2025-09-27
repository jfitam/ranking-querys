# ndcg/__init__.py
from .dcg_score import dcg_score
from .ndcg_score import ndcg_score
from .ndcg_scores_set import ndcg_scores_set

__all__ = ["dcg_score", "ndcg_score", "ndcg_scores_set"]

