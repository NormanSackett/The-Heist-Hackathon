import random
import numpy as np

class node():
    def __init__ (self, weight_arr, bias_arr, activation_func):
        self.weight_arr = weight_arr
        self.bias_arr = bias_arr
        self.activation_func = activation_func

def construct_layers():
    input_layer = np.array([])
    layer_one = np.array([])
    layer_two = np.array([])
    output_layer = np.array([])
    for i in range(1024): # first layer generation for input of 1024 words
        np.append(input_layer,
                  node(np.array([random.random() for i in range(512)]),
                  np.array([random.random() for i in range(512)]),
                  ReLU))
        
def ReLU(x):
    return np.maximum(0, x)

def GELU(x):
    return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def dReLU(x):
    """Derivative of ReLU w.r.t. its input z."""
    return (x > 0).astype(float)


def dGELU(x):
    """Exact-ish derivative of GELU using the erf-based formula.

    Uses the identity: GELU(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
    derivative = 0.5 * (1 + erf(x/sqrt(2))) + (x / sqrt(2*pi)) * exp(-x**2/2)
    """
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0))) + (x * np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))


def init_network(sizes, activation='relu', seed=None):
    """Initialize a fully-connected feed-forward network.

    sizes: list of ints, e.g. [input_dim, hidden1, ..., output_dim]
    activation: 'relu' or 'gelu' (applies to all hidden and output layers)

    Returns: (weights, biases, activations)
      - weights: list of numpy arrays with shape (out, in)
      - biases: list of numpy arrays with shape (out,)
      - activations: list of callables (one per layer)

    This helper is provided because the simple `node` class earlier in the
    file does not conveniently represent a matrix-based dense network.
    """
    if seed is not None:
        np.random.seed(seed)
    weights = [np.random.randn(y, x) * np.sqrt(2.0 / max(1, x)) for x, y in zip(sizes[:-1], sizes[1:])]
    biases = [np.random.randn(y) for y in sizes[1:]]
    if activation == 'relu':
        act = ReLU
        dact = dReLU
    else:
        act = GELU
        dact = dGELU
    activations = [act for _ in sizes[1:]]
    activation_primes = [dact for _ in sizes[1:]]
    return weights, biases, activations, activation_primes


def forward(weights, biases, activations, x):
    """Compute forward pass and return activations and pre-activations (zs).

    - x: 1D array (input vector)
    Returns: (acts, zs)
      - acts: list of layer activations starting with input a0 and ending with a_L
      - zs: list of pre-activation arrays z1..z_L
    """
    a = x
    acts = [a]
    zs = []
    for W, b, act in zip(weights, biases, activations):
        z = W.dot(a) + b
        zs.append(z)
        a = act(z)
        acts.append(a)
    return acts, zs


def mse_loss_derivative(a_out, y):
    """Mean squared error derivative dL/da for output activation a_out."""
    return (a_out - y)


def backward(weights, biases, activation_primes, x, y, lr=1e-2, loss_derivative=None):
    """Perform one step of backpropagation and update weights/biases in-place.

    Arguments:
      - weights, biases: lists of numpy arrays (weights: (out,in), biases: (out,))
      - activation_primes: list of callables computing derivative w.r.t. z per layer
      - x: input vector (shape (in,))
      - y: target vector (shape matching network output)
      - lr: learning rate
      - loss_derivative: callable(a_out, y) -> dL/da_out. If None uses MSE.

    Returns: None (updates weights and biases in-place). For convenience also
    returns the tuple (weights, biases).

    Notes / assumptions:
      - This implements standard dense-layer backpropagation with a simple
        elementwise activation derivative.
      - Shapes: W.dot(a) + b where W is (out, in) and a is (in,)
    """
    if loss_derivative is None:
        loss_derivative = mse_loss_derivative

    acts, zs = forward(weights, biases, [ap for ap in activation_primes], x)
    # output layer index
    L = len(weights) - 1

    # delta for output layer: dL/dz_L = dL/da_L * da_L/dz_L
    a_L = acts[-1]
    dL_da = loss_derivative(a_L, y)
    d_act = activation_primes[L](zs[L])
    delta = dL_da * d_act

    # Backpropagate
    for l in range(L, -1, -1):
        a_prev = acts[l]
        # gradient w.r.t. weights and biases
        dW = np.outer(delta, a_prev)
        db = delta
        # update parameters
        weights[l] = weights[l] - lr * dW
        biases[l] = biases[l] - lr * db

        if l > 0:
            # propagate delta to previous layer
            delta = weights[l].T.dot(delta) * activation_primes[l-1](zs[l-1])

    return weights, biases


# Example usage (doctest-style comment):
# sizes = [3, 5, 2]
# W, b, acts, dacts = init_network(sizes, activation='relu', seed=1)
# x = np.random.randn(3)
# y = np.array([0.5, -0.2])
# W, b = backward(W, b, dacts, x, y, lr=1e-2)
