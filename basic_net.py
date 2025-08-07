'''basic_net.py

Holds class to build a basic CNN network.

MKP - August 2025
'''
import os
import pickle
import tensorflow as tf

import neural_net_base

class CNN(neural_net_base.NeuralNetwork):
    
    def __init__(self, data_shape, output_labels):
        self.data_shape = data_shape
        self.output_labels = output_labels
        self.output_layer_names = ['out_' + label for label in self.output_labels]
        
        self.net = None
        self.params = None
        
    def build_model(self, params):
        self.params = params
        
        self.initialize(params, print_summary=False)
        self.compile(lr=params['lr'])
        
    def initialize(self, params, print_summary=True):
        wt_init_str = params['wt_init']
        
        k_gain = get_kaiming_gain(params['net_act_func'])
        num_wt_layers = count_wt_layers(net_type='cnn',
                                        num_conv_blocks=params['num_conv_blocks'],
                                        num_dense_blocks=params['num_dense_blocks'])
        dropout_prob = params['model']['dropout_rate']
        l1_