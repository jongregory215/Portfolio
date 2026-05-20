"""
stockgrader — Stock Analysis Engine

Grade a single equity ticker across fundamental, technical, and quantitative
dimensions. Produces an overall grade, five portfolio sub-grades, a price
ladder with fair-value estimates, and constructs optimal portfolios for five
risk-tiered model portfolios.

Quick start:
    from stockgrader.models import AnalysisResult, Grade
    from stockgrader.config import get_config
"""

__version__ = "0.1.0"
__author__ = "Jonathan Gregory"
