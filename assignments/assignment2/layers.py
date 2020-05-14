import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength*np.sum(np.square(W))
    grad = 2*np.array(W)*reg_strength
    
    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''   
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    
    if predictions.ndim == 2:
        expons = np.exp(predictions.T - np.max(predictions, 1)).T
        probs = expons / np.sum(expons, 1).reshape(1, -1).T
    elif predictions.ndim == 1:
        expons = np.exp(predictions - np.max(predictions))
        probs = expons / np.sum(expons)
    else:
        raise Exception("Wrong array shape!")

    return probs
    
    
def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value for the entire batch
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    
    return -np.log(np.choose(target_index, probs.T))


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    if predictions.ndim == 1:
        probs[target_index] -= 1 # -1 is the result of algebraic calculations left behind (dCE/dZ, where Z=XW+B)
    else:
        probs[np.arange(probs.shape[0]), target_index] -= 1
    dprediction = probs / len(target_index) # norming is needed to find the mean of grad for all samples in batch

    return loss.mean(), dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X
        return np.maximum(X, 0)
        

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_input = np.ones_like(self.X)
        d_input[np.where(self.X < 0)] = 0
        d_result = d_out * d_input
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        self.value = value
        self.grad = np.zeros_like(value)

        


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        out = np.dot(X, self.W.value) + self.B.value
        return out

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        d_input = np.dot(d_out, (self.W.value).T)
        dW = np.dot(self.X.T, d_out)
        dB = np.sum(d_out, axis=0, keepdims=True)
        self.W.grad = dW
        self.B.grad = dB
        
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
