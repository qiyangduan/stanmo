""" model implementation classes
Each class in model package should implement one mining model as a sub-class of the BaseMiningModel.
"""
__author__ = 'duan'
__model_spec_name__ = 'neuralnet2layer.NeuralNet2Layer'
__model_spec_desc__ = 'Churn prediction by deep learning'

from .neuralnet2layer import NeuralNet2Layer

