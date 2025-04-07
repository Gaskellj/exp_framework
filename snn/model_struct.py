"""
Module for simulating spiking neural networks (SNNs) with spiky neurons.

Authors: Abhay Kashyap, Atharv Tekurkar
"""

import numpy as np
from snn.ring_buffer import RingBuffer

# Constants
SPIKE_DECAY = 0.1
MAX_BIAS = 1
MAX_FIRELOG_SIZE = 200

class SpikyNode:
    """
    Class representing a spiky neuron.
    """

    def __init__(self, size):
        """
        Initializes a spike neuron.

        Parameters:
            size (int): Number of weights plus the bias.
        """
        # a list of weights and a bias (last item in the list)
        self._weights = np.random.uniform(-1, 1, (size + 1))
        self.level = -np.inf  # activation level
        self.firelog = RingBuffer(
            MAX_FIRELOG_SIZE)  # tracks whether the neuron fired or not
        self.levels_log = []

    def compute(self, inputs):
        """
        Compute the neuron's output based on inputs.
        
        Parameters:
            inputs (list): Inputs into this node.

        Returns:
            tuple: (1.0 if the neuron fired, 0.0 otherwise,
                    the neuron's current level).
        """

        self.level *= (1 - SPIKE_DECAY)

        if (len(inputs) + 1) != len(self._weights):
            print(f"Error: {len(inputs)} inputs vs {len(self._weights)} \
                  weights; weights: {self._weights}")
            return 0.0, self.level

        weighted_sum = sum(inputs[i] * self._weights[i]
                           for i in range(len(inputs)))

        if self.level == -np.inf:
            self.level = weighted_sum
        else:
            self.level += weighted_sum

        self.levels_log.append(self.level)

        if self.level >= self.get_bias():
            # print("Fired --> activation level reset to 0.0\n")
            self.level = -np.inf
            self.firelog.add(1)
            return 1.0, self.level
        
        # print("\n")
        self.firelog.add(0)
        return 0.0, self.level

    def duty_cycle(self, window=100):
        """
        Measures how frequently the neuron fires.

        Parameters:
            window (int): How far back to compute the duty cycle.

        Returns:
            float: What pecent of the last `window` timesteps the neuron fired, in decimal form.
        
        """
        if self.firelog.length() == 0:
            return 0.0
        if window is None or window > self.firelog.length():
            window = self.firelog.length()
        recent_fires = self.firelog.get()[-window:]
        # Return an unrounded fraction so the mapping can be more precise.
        return sum(recent_fires) / window

    def set_weights(self, input_weights):
        """
        Sets the neuron's weights.
        
        Parameters:
            input_weights (list): Incoming weights into the neuron.
        """
        if len(input_weights) != len(self._weights):
            print("Weight size mismatch in node")
        else:
            self._weights = input_weights.copy()
            self._weights[:-1] = list(map(lambda x: abs(x), self._weights[:-1])) # Taking the abs of weight?
            # self._weights = input_weights.copy()

    def set_bias(self, val):
        """Sets the neuron's bias."""
        self._weights[-1] = val

    def get_bias(self):
        """Returns the bias from the combined list of weights and bias."""
        return self._weights[-1]

    def print_weights(self):
        """Prints the combined list of weights and bias."""
        print(self._weights)

    def get_levels_log(self):
        """Return the list of the neuron's recent activation levels."""
        return self.levels_log

    @property
    def weights(self):
        """Get the weights of the neuron."""
        return self._weights


class SpikyLayer:
    """
    Collection of multiple neurons (SpikyNodes).
    """

    def __init__(self, num_nodes, num_inputs):
        """
        Initializes a SpikyLayer.

        Parameters:
            num_nodes (int): Number of neurons in the layer.
            num_inputs (int): Number of inputs into each neuron the layer.
        """

        self.nodes = [SpikyNode(num_inputs) for _ in range(num_nodes)]

    def compute(self, inputs):
        """
        Feeds input to each node and returns their output.
        
        Parameters:
            inputs (list): A list of inputs into this layer.

        Returns:
            tuple: (a list of all neuron outputs, a list of all neuron levels)
        """

        outputs = []
        levels = []

        for node in self.nodes:
            output, level = node.compute(inputs)
            outputs.append(output)
            levels.append(level)

        return outputs, levels

    def set_weights(self, input_weights):
        """
        Sets weights for all the neurons in the layer.
        
        Parameters:
            input_weights (list): List of weights for all neurons in the layer.
        """
        if not self.nodes:
            return
        weights_per_node = len(input_weights) // len(self.nodes)
        for idx, node in enumerate(self.nodes):
            start = idx * weights_per_node
            end = start + weights_per_node
            node.set_weights(input_weights[start:end])

    def duty_cycles(self, window=100):
        """
        Returns the duty cycles for the neurons in the layer.
        
        Parameters:
            window (int): How many previous time steps to compute the duty cycle.

        Returns:
            list: The duty cycle for each neuron in the layer.
        """
        return [node.duty_cycle(window) for node in self.nodes]


class SpikyNet:
    """
    Combines 2 spiky layers, a hidden layer and an output layer.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes network.
        
        Parameters:
            input_size (int): Number of inputs into the network.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of outputs.
        """

        self.hidden_layer = SpikyLayer(hidden_size, input_size)
        self.output_layer = SpikyLayer(output_size, hidden_size)

    def compute(self, inputs, firelog_window=100):
        """
        Passes the input through the hidden layer.
        
        Parameters:
            inputs (list): Inputs to the network.
            firelog_window (int): Number of previous timesteps to compute duty cycle.
        """

        hidden_output, hidden_levels = self.hidden_layer.compute(inputs)
        output, levels = self.output_layer.compute(hidden_output)
        return output, levels

    def set_weights(self, input_weights):
        """Assigns weights to the hidden and the output layer."""
        self.hidden_layer.set_weights(input_weights['hidden_layer'])
        self.output_layer.set_weights(input_weights['output_layer'])

    def print_structure(self):
        """Displays the network weights."""
        print("Hidden Layer:")
        for node_index, hidden_node in enumerate(self.hidden_layer.nodes):
            print(f"Node {node_index}: ", end="")
            hidden_node.print_weights()
        print("\nOutput Layer:")
        for node_index, output_node in enumerate(self.output_layer.nodes):
            print(f"Node {node_index}: ", end="")
            output_node.print_weights()
        print("\n")
