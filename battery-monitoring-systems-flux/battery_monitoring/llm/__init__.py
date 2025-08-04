"""
LLM module for battery monitoring system.

This module provides LLM-powered chatbot capabilities and model evaluation.
"""

from .chatbot import BatteryChatbot
from .evaluator import LLMEvaluator

# DeepEval integration for model evaluation
try:
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancy, Faithfulness, ContextRelevancy
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    print("Warning: DeepEval not available. Install with: pip install deepeval")

__all__ = ['BatteryChatbot', 'LLMEvaluator', 'DEEPEVAL_AVAILABLE'] 