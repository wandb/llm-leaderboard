"""
BFCL Package
This is the root package for BFCL (Benchmark for Function Calling LLMs).
"""

from .bfcl._llm_response_generation import get_args, main as generation_main
from .bfcl.eval_checker.eval_runner import main as evaluation_main

__all__ = ['get_args', 'generation_main', 'evaluation_main'] 