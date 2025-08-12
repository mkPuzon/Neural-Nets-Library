'''utils.py

MKP - August  2025
'''
import os
import yaml
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime

def examine_dict(data_dict, title="Examining Dictionary Object"):
    """
    Neatly prints the structure, types, and sample data of a dictionary.
    
    Args:
        data_dict (dict): The dictionary to examine.
        title (str): An optional title for the output header.
    """
    
    def _print_level(data, indent_str="  ", level=0):
        """Recursive helper function to print each level of the dictionary."""
        indent = indent_str * level
        
        # Handle dictionaries
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"{indent}├─ {key} (dict):")
                    _print_level(value, indent_str, level + 1)
                elif isinstance(value, list):
                    list_info = f"list, {len(value)} items"
                    if value:
                        list_info += f" of type {type(value[0]).__name__}"
                    print(f"{indent}├─ {key} ({list_info}):")
                    # If the list contains dictionaries, examine the first one as a sample
                    if value and isinstance(value[0], dict):
                        _print_level(value[0], indent_str, level + 2)
                else:
                    display_val = repr(value)
                    if len(display_val) > 60:
                        display_val = display_val[:57] + "..."
                    print(f"{indent}├─ {key}: {display_val}")

    # --- Main Function Body ---
    print("\n"+ "=" * 70)
    print(f"{title}".center(70))
    print("=" * 70)

    if not isinstance(data_dict, dict):
        print("Input is not a valid dictionary.".ljust(51))
        print("=" * 70)
        return

    print(f"Top-level keys: {len(data_dict):<33}")
    print("-" * 70)
    _print_level(data_dict, level=1)
    print("=" * 70 + "\n")
    
def get_net_params(filename, config_folder='network_params'):
    """
    Load neural network hyperparameters from a YAML file.
    
    Args:
        filename (str): Name of the YAML file (with or without .yaml/.yml extension).
        config_folder (str): Subfolder where config files are stored.

    Returns:
        dict: Dictionary of parameters.
    """
    if not (filename.endswith('.yaml') or filename.endswith('.yml')):
        filename += '.yaml'

    filepath = os.path.join(config_folder, filename)

    with open(filepath, "r") as f:
        params = yaml.safe_load(f)

    return params

def get_data_params(filename, config_folder='dataset_params'):
    """
    Load data hyperparameters from a YAML file.
    
    Args:
        filename (str): Name of the YAML file (with or without .yaml/.yml extension).
        config_folder (str): Subfolder where config files are stored.

    Returns:
        dict: Dictionary of parameters.
    """
    if not (filename.endswith('.yaml') or filename.endswith('.yml')):
        filename += '.yaml'

    filepath = os.path.join(config_folder, filename)

    with open(filepath, "r") as f:
        params = yaml.safe_load(f)

    return params

def analyze_history(history, plot=True):
    
    """
    Analyze the performance of a model during training, including the number of epochs, 
    best training and validation performance, and optional plots of the training and validation curves.

    Args:
        history (keras.src.callbacks.history.History): The history object returned by the fit method of the model.
        plot (bool): Whether to plot the training and validation curves. Defaults to True.

    Returns:
        None
    """
    print("="*60)
    print("MODEL HISTORY".center(60))
    print("="*60)
    
    print(f"Ran for {len(history.epoch)} epochs out of a max of {history.params['epochs']} epochs.")
    print(f"History tracking: {[key for key in history.history.keys()]}\n") # ['accuracy', 'loss', 'val_accuracy', 'val_loss']
    
    metrics = [m for m in history.history.keys() if not m.startswith('val_')]
    
    for metric in metrics:
        # Get training and validation values
        train_values = history.history[metric]
        val_metric = 'val_' + metric
        val_values = history.history.get(val_metric)

        # Find the best performance
        if 'acc' in metric or 'iou' in metric or 'auc' in metric: # Metrics to maximize
            best_epoch_train = np.argmax(train_values)
            best_val = np.max(val_values) if val_values else None
            best_epoch_val = np.argmax(val_values) if val_values else None
        else: # Metrics to minimize (like loss)
            best_epoch_train = np.argmin(train_values)
            best_val = np.min(val_values) if val_values else None
            best_epoch_val = np.argmin(val_values) if val_values else None
            
        print(f"Metric: {metric.replace('_', ' ').title()}")
        print(f"  - Best Train:   {train_values[best_epoch_train]:.4f} at epoch {best_epoch_train + 1}")
        if val_values:
            print(f"  - Best Val: {best_val:.4f} at epoch {best_epoch_val + 1}")
    
    print("="*60 + "\n")
    
    # create two side by side plots: one of train/val loss and one of train/val accuracy
    if plot:
        num_metrics = len(metrics)
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
        
        if num_metrics == 1: # Matplotlib returns a single Axes object if only 1 subplot
            axs = [axs]

        for i, metric in enumerate(metrics):
            val_metric = 'val_' + metric
            
            # Plot training and validation curves
            axs[i].plot(history.history[metric], label=f'Train {metric}', color='dodgerblue', lw=2)
            if val_metric in history.history:
                axs[i].plot(history.history[val_metric], label=f'Val {metric}', color='darkorange', lw=2)

                # Highlight the best epoch for validation
                if 'acc' in metric or 'iou' in metric or 'auc' in metric:
                    best_epoch = np.argmax(history.history[val_metric])
                else:
                    best_epoch = np.argmin(history.history[val_metric])
                
                axs[i].axvline(best_epoch, linestyle='--', color='gray', 
                            label=f'Best Val at {best_epoch + 1}')

            axs[i].set_title(f'{metric.replace("_", " ").title()} Over Epochs', fontsize=14)
            axs[i].set_xlabel('Epoch', fontsize=12)
            axs[i].set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            axs[i].legend()
            
        plt.tight_layout()
        plt.show()
        
def create_run_directory(model_name="model", data_name="data"):
    """
    Creates a directory to store model and data information.

    Args:
        model_name (str): The name of the neural network.
        data_name (str): The name of the data set used to train the model.

    Returns:
        str: The path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir_name = f"{model_name}-{data_name}_{timestamp}"
    run_path = os.path.join("training_runs", run_dir_name)
    
    os.makedirs(run_path, exist_ok=True)
    
    print(f"Model stored in directory: {run_path}")
    
    return run_path
    
    
def load_json_results(name, base_path="/home/mkpuzon/code/Neural-Nets-Library/training_runs/", print_results=False):
    
    with open(os.path.join(base_path, name, "results.json"), 'r') as f:
        results = json.load(f)
        
    if print_results:
        examine_dict(results, title=f"Info for {name}")
        
    return results

def create_callbacks_from_config(params, verbose=False):
    if 'callbacks' not in params:
        if verbose:
            print("No callback configurations found in params.")
        return []

    CALLBACK_MAP = {
        'early_stopping': tf.keras.callbacks.EarlyStopping,
        'model_checkpoint': tf.keras.callbacks.ModelCheckpoint,
        'reduce_lr_on_plateau': tf.keras.callbacks.ReduceLROnPlateau,
        'csv_logger': tf.keras.callbacks.CSVLogger
    }
    
    callbacks_list = []
    config = params['callbacks']

    for key, callback_params in config.items():
        if key in CALLBACK_MAP:
            # --- Automatically create directories for file paths ---
            for path_key in ['filepath', 'filename']:
                if path_key in callback_params:
                    # Get the directory part of the path
                    dir_name = os.path.dirname(callback_params[path_key])
                    if dir_name: # Ensure it's not empty
                        os.makedirs(dir_name, exist_ok=True)

            # --- Create and append the callback ---
            callback_class = CALLBACK_MAP[key]
            callbacks_list.append(callback_class(**callback_params))
            
            if verbose:
                print(f"Configured {key} with params: {callback_params}")
        else:
            # Warn about unknown keys (potential typos)
            if verbose:
                print(f"[WARNING] Unknown callback key '{key}' found in config. Ignoring.")
    
    return callbacks_list

def get_optimizer(params):
        opt_config = params["optimizer"]
        if opt_config["type"].lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=opt_config["learning_rate"])
        elif opt_config["type"].lower() == "adamw":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=opt_config["learning_rate"])
        else:
            raise NotImplementedError(f"Optimizer {opt_config['type']} not implemented.")
        
        return optimizer

def get_model_type(model_name):
    if "cnn" in model_name.lower():
        return 'cnn'
    elif "mlp" in model_name.lower():
        return 'mlp'
    else:
        raise ValueError(f"Unknown model type: {model_name}")