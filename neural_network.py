"""Learning and prediction functions for artificial neural networks."""

import collections
import common
import math
import random
import sys

# Throughout this file, layer 0 of a neural network is the inputs, layer 1
# is the first hidden layer, etc.; the last layer is the outputs.

class NeuralNetwork:
  """An artificial neural network.

  Fields:
    weights: a list of lists of lists of numbers, where
       weights[a][b][c] is the weight into unit b of layer a+1 from unit c in
         layer a
    num_hidden_layers: an integer, the number of hidden layers in the network
  """

  def __init__(self, weights=None):
    self.weights = weights
    if weights:
      self.num_hidden_layers = len(weights) - 1

  def get_unit_values(self, features):
    """Calculate the activation of each unit in a neural network.

    Args:
      features: a vector of feature values

    Returns:
      units, a list of lists of numbers, where
        units[a][b] is the activation of unit b in layer a
    """
    # COMPLETE THIS IMPLEMENTATION
    self.units = []
    self.units.append (features)
    for i in xrange (len (self.weights)):
      self.units.append ([])

    for layer in xrange (1, len (self.weights) + 1):
      for unitWeights in self.weights[layer - 1]:
         self.units[layer].append (self.activation (common.dot (unitWeights, self.units[layer - 1])))
    return self.units


  def predict(self, features):
    """Calculate the prediction of a neural network on one example

    Args:
      features: a vector of feature values

    Returns:
      A list of numbers, the predictions for each output of the network
          for the given example.
    """
    # COMPLETE THIS IMPLEMENTATION
    return [0]


  def calculate_errors(self, unit_values, outputs):
    """Calculate the backpropagated errors for an input to a neural network.

    Args:
      unit_values: unit activations, a list of lists of numbers, where
        unit_values[a][b] is the activation of unit b in layer a
      outputs: a list of correct output values (numbers)

    Returns:
      A list of lists of numbers, the errors for each hidden or output unit.
          errors[a][b] is the error for unit b in layer a+1.
    """
    # COMPLETE THIS IMPLEMENTATION
    return [[0]]

  def activation(self, v):
    return 1 / (1 + math.exp(-v))

  def learn(self,
      data,
      num_hidden=16,
      max_iterations=1000,
      learning_rate=1,
      num_hidden_layers=1):
    """Learn a neural network from data.

    Sets the weights for a neural network based on training data.

    Args:
      data: a list of pairs of input and output vectors, both lists of numbers.
      num_hidden: the number of hidden units to use.
      max_iterations: the max number of iterations to train before stopping.
      learning_rate: a scaling factor to apply to each weight update.
      num_hidden_layers: the number of hidden layers to use.
        Unless you are doing the extra credit, you can ignore this parameter.

    Returns:
      This object, once learned.
    """
    # COMPLETE THIS IMPLEMENTATION
    # Use predict, get_unit_values, and calculate_errors
    # in your implementation!

    return self
