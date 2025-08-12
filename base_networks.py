'''cnn_basic.py

Holds class to build a basic CNN network.

MKP - August 2025
'''
import os
import time
import yaml

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from neural_net_base import NeuralNetwork
from utils import create_callbacks_from_config, get_optimizer, create_run_directory

class CNN(NeuralNetwork):
    def __init__(self, params):
        self.params = params
        self.name = params.get('model_name', 'cnn')
        self.net = None
        print(f"Initialized model '{self.name}'.")

    def build_model(self, data_params, print_summary=False):
        self.data_params = data_params
        
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

        self.net = tf.keras.Model(inputs=input_layer, outputs=outputs, name=self.name)
        
        if print_summary:
            self.net.summary()
            
    def fit_model(self, x_train, y_train, x_val, y_val, save_wts=False, verbose=False):
        optimizer = get_optimizer(self.params)
        
        self.net.compile(
            optimizer=optimizer,
            loss=self.params["loss"],
            metrics=self.params["metrics"]
        )
        
        if verbose:
            self.net.summary()
            
        # ---- Callbacks ---- #
        callbacks_list = create_callbacks_from_config(self.params, verbose=verbose)
        
        # ---- Training ---- # 
        st = time.time()
        history = self.net.fit(
            x_train, y_train,
            validation_data=(x_val, y_val) if x_val is not None else None,
            epochs=self.params["max_epochs"],
            batch_size=self.data_params["minibatch_sz"],
            shuffle=self.data_params.get("shuffle", True),
            callbacks=callbacks_list,
            verbose=verbose
        )
        training_duration_seconds = time.time() - st
        if verbose:
            print(f"Training took {training_duration_seconds:.2f} seconds.")
        
        if save_wts:
            self.save_full(data_name=self.data_params["dataset_name"], history=history, training_duration_seconds=training_duration_seconds)
            
        return history, training_duration_seconds
        
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
    
    def visualize_conv_layers(
        self,
        test_data,
        index=0,
        match_str='conv2d',
        neurons=None):
        """
        Visualize specific Conv2D layer feature maps for a given test image with a shared
        colormap per layer based on the selected neurons.
        
        Args:
            test_data (np.ndarray): Test dataset (images).
            index (int): Index of the image to visualize.
            match_str (str): String to match Conv2D layer names. Defaults to 'conv2d'.
            neurons (list[int] | None): List of neuron/filter indices to visualize. 
                                        If None, all neurons will be shown.
        """
        if self.net is None:
            raise ValueError("Model is not built. Build and train before visualization.")

        test_img = test_data[index:index+1]

        conv_layer_names = [layer.name for layer in self.net.layers if match_str in layer.name]
        conv_outputs = [layer.output for layer in self.net.layers if match_str in layer.name]

        if not conv_outputs:
            print(f"No layers found matching '{match_str}'.")
            return

        activation_model = tf.keras.Model(inputs=self.net.input, outputs=conv_outputs)
        activations = activation_model.predict(test_img)

        for layer_name, feature_map in zip(conv_layer_names, activations):
            num_filters = feature_map.shape[-1]
            size = feature_map.shape[1]
            selected_neurons = neurons if neurons is not None else range(num_filters)

            # Compute shared vmin/vmax for the subset
            subset_maps = feature_map[0, :, :, selected_neurons]
            vmin = subset_maps.min()
            vmax = subset_maps.max()

            n_cols = min(6, len(selected_neurons))
            n_rows = int(np.ceil(len(selected_neurons) / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
            axes = np.array(axes).reshape(-1)

            fig.suptitle(f"Layer: {layer_name}", fontsize=14, y=1.02)

            for ax, neuron_idx in zip(axes, selected_neurons):
                fm = feature_map[0, :, :, neuron_idx]
                im = ax.imshow(fm, cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f"Neuron {neuron_idx}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            # Hide unused subplots
            for ax in axes[len(selected_neurons):]:
                ax.axis("off")
                
            # Add colorbar to the right of all subplots without overlapping
            fig.subplots_adjust(right=0.88)  # make space for colorbar
            cbar_ax = fig.add_axes([1.05, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax)
        
            plt.tight_layout()
            plt.show()

    def _save_wts(self, run_path):
        
        weights_path = os.path.join(run_path, f"{self.name}.weights.h5")
        self.net.save_weights(weights_path)
        print(f"Saved model weights to: {weights_path}")
        
    def save_full(self, data_name, history, training_duration_seconds):
        """
        Saves the model weights (.h5) and run metadata (.yaml) including:
        - Dataset name
        - Epochs trained
        - Training duration
        - Final loss
        - Full training history
        - Path to hyperparameters file (self.params)
        """
        run_path = create_run_directory(model_name=self.name, data_name=data_name)

        self._save_wts(run_path=run_path)

        history_dict = {
            key: [float(i) for i in val]
            for key, val in history.history.items()
        }

        results = {
            "dataset": data_name,
            "params_file": self.params,  # path to .yaml with hyperparameters
            "epochs_trained": len(history_dict.get("loss", [])),
            "training_duration_seconds": round(training_duration_seconds, 2),
            "final_loss": history_dict.get("loss", [None])[-1],
            "history": history_dict
        }

        metadata_path = os.path.join(run_path, "metadata.yaml")
        with open(metadata_path, "w") as f:
            yaml.safe_dump(results, f, sort_keys=False)
        print(f"Saved training metadata to: {metadata_path}")
        
class MLP(NeuralNetwork):
    def __init__(self, params):
        self.params = params
        self.name = params.get('model_name', 'mlp')
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
    