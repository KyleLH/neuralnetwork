"""Unit tests for assignment 4 Extra Credit."""

# THESE TESTS WILL FAIL IF YOU ARE NOT DOING THE EXTRA CREDIT.
# That failure will not count against your score unless you are doing the extra
# credit.

import random
import unittest
from neural_network import NeuralNetwork

XOR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)]

OR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1)]

def make_multiple_outputs(data):
  return [(features, [output]) for features, output in data]

def clip(v):
  """Convert a neural network output to a 0/1 prediction."""
  return 0 if v < 0.5 else 1

class Test(unittest.TestCase):
  longMessage=True

  _MODEL_TWO_HIDDEN_LAYERS = NeuralNetwork([
          [
          # weights into hidden unit 0, first hidden layer
            [0.3, 0.4, 0.5],
          # weights into hidden unit 1, first hidden layer
            [-0.5, -0.6, 0.7],
          ],
          [
          # weights into hidden unit 0, second hidden layer
            [0.1, 0.7],
          # weights into hidden unit 1, second hidden layer
            [-0.2, -0.8],
          # weights into hidden unit 2, second hidden layer
            [-0.3, -0.5],
          ],
          # weights into output unit
          [[0.1, 0.2, 0.9]]])

  def setUp(self):
    random.seed(5)

  def test_get_unit_values(self):
    inputs = [1, 2, -1]
    unit_values = self._MODEL_TWO_HIDDEN_LAYERS.get_unit_values(inputs)
    self.assertEqual(len(unit_values), 4)  # four layers of units
    self.assertEqual(unit_values[0], inputs)  # layer 0 is inputs
    self.assertEqual(len(unit_values[1]), 2)  # layer 1 is hidden units
    self.assertAlmostEqual(unit_values[1][0], 0.6457, places=4)
    self.assertAlmostEqual(unit_values[1][1], 0.0832, places=4)
    self.assertEqual(len(unit_values[2]), 3)  # layer 2 is also hidden units
    self.assertAlmostEqual(unit_values[2][0], 0.5307, places=4)
    self.assertAlmostEqual(unit_values[2][1], 0.4512, places=4)
    self.assertAlmostEqual(unit_values[2][2], 0.4414, places=4)
    self.assertEqual(len(unit_values[3]), 1)  # layer 3 is output units
    self.assertAlmostEqual(unit_values[3][0], 0.6320, places=4)

  def test_calculate_errors(self):
    unit_values = self._MODEL_TWO_HIDDEN_LAYERS.get_unit_values([1, 2, -1])
    errors = self._MODEL_TWO_HIDDEN_LAYERS.calculate_errors(unit_values, [0])
    self.assertEqual(len(errors), 3)  # First hidden, second hidden, output
    self.assertEqual(len(errors[2]), 1)  # One error per output unit
    self.assertAlmostEqual(errors[2][0], -0.1470, places=4)
    self.assertEqual(len(errors[1]), 3)  # One error per hidden unit, second layer
    self.assertAlmostEqual(errors[1][0], -0.0037, places=4)
    self.assertAlmostEqual(errors[1][1], -0.0073, places=4)
    self.assertAlmostEqual(errors[1][2], -0.0326, places=4)
    self.assertEqual(len(errors[0]), 2)  # One error per hidden unit, first layer
    self.assertAlmostEqual(errors[0][0], 0.0025, places=4)
    self.assertAlmostEqual(errors[0][1], 0.0015, places=4)

  def test_learn_or(self):
    data = make_multiple_outputs(OR)
    model = NeuralNetwork().learn(data, num_hidden_layers=2)
    for x, (y,) in data:
      self.assertEqual(clip(model.predict(x)[0]), y,
          msg='datum %s, %s' % (x, y))

  def test_learn_xor(self):
    data = make_multiple_outputs(XOR)
    model = NeuralNetwork().learn(data, num_hidden_layers=2, max_iterations=2000)
    for x, (y,) in data:
      self.assertEqual(clip(model.predict(x)[0]), y,
          msg='datum %s, %s' % (x, y))


if __name__ == '__main__':
  unittest.main()
