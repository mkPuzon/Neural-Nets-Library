'''cnn_basic.py

Holds class to build a basic CNN network.

MKP - August 2025
'''
import tensorflow as tf

from neural_net_base import NeuralNetwork

class CNN(NeuralNetwork):
    def __init__(self, net_params):
        self.params = net_params
        self.name = self.params.get('model_name', 'cnn')
        self.net = None
        print(f"Initialized model '{self.name}'.")

    def build_model(self, data_params, print_summary=False):
        input_layer = tf.keras.Input(shape=data_params['input_shape'])
        x = input_layer

        # ---- Conv/Dense Blocks ---- #
        for layer_cfg in self.params.get('conv_layers', []):
            x = self.add_conv2d_block(x, layer_cfg)
        
        x = tf.keras.layers.Flatten()(x)
        
        for layer_cfg in self.params.get('dense_layers', []):
            x = self.add_dense_block(x, layer_cfg)
            
        # ---- Output Layers ---- #
        if 'output_layer' in self.params:
            print("Building single-output classification head...")
            outputs = self.build_single_output(x, data_params)
        elif 'multi_output_layers' in self.params:
            print("Building multi-output regression head...")
            outputs = self.build_multi_output(x, data_params)
        else:
            raise ValueError("Model config must define either 'output_layer' or 'multi_output_layers'")

        self.net = tf.keras.Model(inputs=input_layer, outputs=outputs)
        
        if print_summary:
            self.net.summary()
            
    # -------- HELPER FUCTIONS -------- #
    def add_conv2d_block(self, x, layer_cfg):
        x = tf.keras.layers.Conv2D(
            filters=layer_cfg['filters'],
            kernel_size=layer_cfg['kernel_size'],
            padding='same'
        )(x)
        if layer_cfg.get('batch_norm', False):
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(layer_cfg['activation'])(x)
        if layer_cfg.get('max_pool', False):
            x = tf.keras.layers.MaxPooling2D()(x)
        return x

    def add_dense_block(self, x, layer_cfg):
        x = tf.keras.layers.Dense(layer_cfg['units'], activation=layer_cfg['activation'])(x)
        if 'dropout_rate' in layer_cfg:
            x = tf.keras.layers.Dropout(layer_cfg['dropout_rate'])(x)
        return x

    def build_single_output(self, x, data_params):
        """Builds a single output layer for classification."""
        output_cfg = self.params['output_layer']
        return tf.keras.layers.Dense(
            units=data_params['num_classes'],
            activation=output_cfg['activation'],
            name='output'
        )(x)
        
    def build_multi_output(self, x, data_params):
        """Builds multiple output layers for regression."""
        output_layers = []
        for label in data_params['output_labels']:
            output_name = f"out_{label}"
            # Each output is a Dense layer with 1 unit for regression
            output_layers.append(tf.keras.layers.Dense(units=1, name=output_name)(x))
        return output_layers
    
class MLP(NeuralNetwork):
    def __init__(self, net_params):
        self.params = net_params
        self.name = self.params.get('model_name', 'mlp')
        self.net = None
        print(f"Initialized model '{self.name}'.")

    def build_model(self, data_params, print_summary=False):
        input_layer = tf.keras.Input(shape=data_params['input_shape'])
        x = input_layer
        
        for layer_cfg in self.params.get('dense_layers', []):
            x = self.add_dense_block(x, layer_cfg)
            
        # ---- Output Layers ---- #
        if 'output_layer' in self.params:
            print("Building single-output classification head...")
            outputs = self.build_single_output(x, data_params)
        elif 'multi_output_layers' in self.params:
            print("Building multi-output regression head...")
            outputs = self.build_multi_output(x, data_params)
        else:
            raise ValueError("Model config must define either 'output_layer' or 'multi_output_layers'")

        self.net = tf.keras.Model(inputs=input_layer, outputs=outputs)
        
        if print_summary:
            self.net.summary()
            
    # -------- HELPER FUCTIONS -------- #
    def add_dense_block(self, x, layer_cfg):
        x = tf.keras.layers.Dense(layer_cfg['units'], activation=layer_cfg['activation'])(x)
        if 'dropout_rate' in layer_cfg:
            x = tf.keras.layers.Dropout(layer_cfg['dropout_rate'])(x)
        return x

    def build_single_output(self, x, data_params):
        output_cfg = self.params['output_layer']
        return tf.keras.layers.Dense(
            units=output_cfg['units'],
            activation=output_cfg['activation'],
            name='output'
        )(x)
        
    def build_multi_output(self, x, data_params):
        """Builds multiple output layers for regression."""
        output_layers = []
        for label in data_params['output_labels']:
            output_name = f"out_{label}"
            # Each output is a Dense layer with 1 unit for regression
            output_layers.append(tf.keras.layers.Dense(units=1, name=output_name)(x))
        return output_layers
    