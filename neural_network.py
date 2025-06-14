'''networks.py

The base parent class for Neural Network objects.

MKP - June 2025
'''
import os

class NeuralNetwork:
    
    def __init__(self):
        pass
    
    def build_model(params):
        pass
    
    def initialize(self):
        pass
    
    def compile(self, lr=1e-3):
        pass
    
    def save_wts(self, path='wts', filename='model'):
        dst = os.path.join(path, filename)
        pass
        
    def load_wts(self, path='wts', filename='model'):
        src = os.path.join(path, filename)
        pass
    
    def get_layer_names(self, exclude_input_layer=True, exclude_output_layer=True):
        pass
    
    def get_num_layers(self, exclude_input_layer=True, exclude_output_layer=True):
        pass
    
    def get_layer_activations(self, x, split, match_str=None, exclude_input_layer=True, exclude_output_layer=True):
        pass
    
    def fit(self, data, max_epochs=1000, verbose=2, patience=10, minibatch_sz=None):
        pass
    
    def predict(self, x_test):
        pass