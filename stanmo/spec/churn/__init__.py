""" model implementation classes
Each class in model package should implement one mining model as a sub-class of the BaseMiningModel.
"""
__author__ = 'duan'
__model_spec_name__ = 'churn.ChurnMiningModel'
__model_spec_desc__ = 'Churn Prediction for Telco'
from .churnmodelspec import ChurnMiningModel
# from .churnmodelflask import ChurnModelFlask
