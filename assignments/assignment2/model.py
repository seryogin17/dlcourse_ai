import numpy as np
from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.hidden_layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.hidden_layer2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu_layer1 = ReLULayer()
        self.relu_layer2 = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        
        # Hint: using self.params() might be useful!
        
        params = self.params()
        for param_key in params:
            params[param_key].grad = 0
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        
        hid_1 = self.hidden_layer1.forward(X)
        relu_1 = self.relu_layer1.forward(hid_1)
        hid_2 = self.hidden_layer2.forward(relu_1)
        loss, dprediction = softmax_with_cross_entropy(hid_2, y)

        d_out_hid_2 = self.hidden_layer2.backward(dprediction)
        d_out_relu1 = self.relu_layer1.backward(d_out_hid_2)
        self.hidden_layer1.backward(d_out_relu1)
        
        
        for param_key in params:
            reg_loss, reg_grad = l2_regularization(params[param_key].value, self.reg)
            loss += reg_loss
            params[param_key].grad += reg_grad
                    
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        hid_1 = self.hidden_layer1.forward(X)
        relu_1 = self.relu_layer1.forward(hid_1)
        hid_2 = self.hidden_layer2.forward(relu_1)
        output = self.relu_layer2.forward(hid_2)
        
        pred = np.argmax(output, axis = 1)
        
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        hidden_layer_params1 = self.hidden_layer1.params()
        for param_key in hidden_layer_params1:
            result[param_key + '-1'] = hidden_layer_params1[param_key]
            
        hidden_layer_params2 = self.hidden_layer2.params()
        for param_key in hidden_layer_params2:
            result[param_key + '-2'] = hidden_layer_params2[param_key]

        return result
