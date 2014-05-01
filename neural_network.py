"""Learning and prediction functions for artificial neural networks."""

# NAME: Kyle Holzinger
# BUID: U92663004
# DATE: 9.4.14

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
    unit_values = self.get_unit_values (features)
    return unit_values[len (unit_values)-1]


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

    ''' 
      as a note, for this implementation I flipped around unit_values and errors, so as to 
      make it easier iterating through them. Errors is flipped back when returned.
    '''


    errors = []
    unit_values = unit_values[::-1]
    for i in xrange (len (unit_values) - 1):
      errors.append ([])

    # calculate error of each node in the outputs.
    # this algorithm is different than for the hidden layers.
    for out in xrange (len (unit_values[0])):
      err = unit_values[0][out] * (1 - unit_values[0][out]) * (outputs[out] - unit_values[0][out])
      errors[0].append (err)
    errors = errors[::-1]
    unit_values = unit_values[::-1]


    # calculate error of each node in the hidden layers.
    for layer in xrange (len (unit_values) - 2, 0, -1):
      for node in xrange (len (unit_values[layer])):
        err = 0
        cur_val = unit_values[layer][node]
        for parent in xrange (len (unit_values[layer+1])):
          weight = self.weights[layer][parent][node]
          parent_err = errors[layer][parent]

          err += cur_val * (1 - cur_val) * (weight * parent_err)
        errors[layer-1].append (err)
    return errors

  def activation(self, v):
    return 1 / (1 + math.exp(-v))

  def learn(self,
      data,
      num_hidden=7,
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

    cur_iteration = 0
    self.weights = [[]]

    # set up weights for first hidden layer
    for j in xrange (num_hidden):
      weight_list = []
      for i in xrange (len (data[0][0])):
          weight_list.append (random.random())
      self.weights[0].append (weight_list)

    # set up weights for other layers
    for layer in xrange (1, num_hidden_layers + 1):
      self.weights.append ([])
      # set up weights for non-last layer
      if layer < num_hidden_layers:
        for i in xrange (num_hidden):
          weight_list = []
          for j in xrange (num_hidden):
            weight_list.append (random.random())
          self.weights[layer].append (weight_list)
      # set up for last layer
      else: 
        for i in xrange (len (data[0][1])):
          weight_list = []
          for j in xrange (num_hidden):
            weight_list.append (random.random ())
          self.weights[layer].append (weight_list)


    while cur_iteration < max_iterations:
      for input, out in data:
        unit_values = self.get_unit_values (input)
        errors = self.calculate_errors (unit_values, out)
        for layer in xrange (1, len (self.weights ) + 1):
          for parent in xrange (len (self.weights[layer-1])):
            for child in xrange (len (self.weights[layer-1][parent])):
              self.weights[layer-1][parent][child] = self.weights[layer-1][parent][child] + (learning_rate * unit_values[layer-1][child] * errors[layer-1][parent])
      cur_iteration += 1


    return self