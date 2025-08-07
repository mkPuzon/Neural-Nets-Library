'''neural_net_base.py

The base parent class for Neural Network objects.

MKP - June 2025
'''
import os

class NeuralNetwork:
    def __init__(self):
        pass

    def build_model(self, params):
        pass

    def initialize(self):
        pass

    def compile(self, lr=1e-3):
        pass

    def save_wts(self, path='wts', filename='model'):
        dst = os.path.join(path, filename)
        self.net.save_weights(dst)

    def load_wts(self, path='wts', filename='model'):
        # Use .index file as proxy for whether wt file exists
        src_index_path = os.path.join(path, '.'.join([filename, 'index']))

        if not os.path.isfile(src_index_path):
            return False

        src = os.path.join(path, filename)
        self.net.load_weights(src).expect_partial()
        return True

    def get_layer_names(self, exclude_input_layer=True, exclude_output_layer=True):
        pass

    def get_num_layers(self, exclude_input_layer=True, exclude_output_layer=True):
        pass

    def get_layer_activations(self, x, split, match_str=None, exclude_input_layer=True, exclude_output_layer=True):
        pass

    def fit(self, data, max_epochs=5000, verbose=2, patience=10, minibatch_sz=None):
        pass

    def predict(self, x_test):
        pass
