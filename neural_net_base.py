'''neural_net_base.py

The base parent class for Neural Network objects.

MKP - June 2025
'''
import os
import json
import tensorflow as tf
from datetime import datetime

class NeuralNetwork:
    def __init__(self):
        pass

    def build_model(self, data_params, print_summary):
        pass

    def compile_model(self):
        if not self.net:
            raise ValueError("Model must be built before compiling. Call .build_model() first.")

        opt_cfg = self.params['optimizer']
        if opt_cfg['type'].lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=opt_cfg['learning_rate'])
        elif opt_cfg['type'].lower() == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(learning_rate=opt_cfg['learning_rate'])
        else:
            raise NotImplementedError(f"Optimizer {opt_cfg['type']} not implemented.")
        
        self.net.compile(
            optimizer=optimizer,
            loss=self.params['loss'],
            metrics=self.params.get('metrics', [])
        )
        print("Model compiled successfully.")

    def save_model_wts(self, data_name, training_duration_seconds):
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_dir_name = f"{self.model.name}-{data_name}_{timestamp}"
        run_path = os.path.join("training_runs", run_dir_name)
        
        os.makedirs(run_path, exist_ok=True)
        
        print(f"Model stored in directory: {run_path}")
            
        # Prepare history dictionary for JSON serialization
        history_dict = {key: [float(i) for i in val] for key, val in self.history.history.items()}

        # Save training results as a JSON file
        results = {
            "dataset": data_name,
            "epochs_trained": len(history_dict['loss']),
            "training_duration_seconds": round(training_duration_seconds, 2),
            "final_loss": history_dict['loss'][-1],
            "history": history_dict
        }

        results_path = os.path.join(run_path, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved training results to: {results_path}")
        
    def load_json_results(self, base_path="/home/mkpuzon/code/Neural-Nets-Library/training_runs/", print_results=False):
        
        with open(os.path.join(base_path, name, "results.json"), 'r') as f:
            results = json.load(f)
            
        if print_results:
            examine_dict(results, title=f"Info for {name}")
            
        return results

    def load_wts(self, base_path="/home/mkpuzon/code/Neural-Nets-Library/training_runs/"):
        path = os.path.join(base_path, self.model.name)
        
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

    def fit(self, verbose):
        pass

    def predict(self, x_test):
        pass
