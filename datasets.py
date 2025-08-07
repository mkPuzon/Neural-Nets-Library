'''datasets.py

https://github.com/tensorflow/datasets

MKP - August 2025
'''
import tensorflow as tf
import tensorflow_datasets as tfds

def load_dataset(dataset_name, params, verbose=False):
    
    if dataset_name.lower() == "mnist":
        if verbose:
            print("[NOTICE] Using MNIST dataset from TensorFlow Datasets, params file will be ignored.")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_val = x_train[50000:]
        y_val = y_train[50000:]
        x_train = x_train[:50000]
        y_train = y_train[:50000]
        
        assert x_train.shape == (50000, 28, 28)
        assert x_val.shape == (10000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
        
        assert y_train.shape == (50000,)
        assert y_val.shape == (10000,)
        assert y_test.shape == (10000,)
    elif dataset_name.lower() == "cifar10":
        if verbose:
            print("[NOTICE] Using CIFAR-10 dataset from TensorFlow Datasets, params file will be ignored.")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_val = x_train[40000:]
        y_val = y_train[40000:]
        x_train = x_train[:40000]
        y_train = y_train[:40000]
        
        assert x_train.shape == (40000, 32, 32, 3)
        assert x_val.shape == (10000, 32, 32, 3)
        assert x_test.shape == (10000, 32, 32, 3)
        
        assert y_train.shape == (40000, 1)
        assert y_val.shape == (10000, 1)
        assert y_test.shape == (10000, 1)
    elif dataset_name.lower() == 'imdb':
        if verbose:
            print("[NOTICE] Using IMDB dataset from TensorFlow Datasets, params file will be ignored.")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
        x_val = x_train[15000:]
        y_val = y_train[15000:]
        x_train = x_train[:15000]
        y_train = y_train[:15000]
        
        assert len(x_train) == 15000
        assert len(x_val) == 10000
        assert len(x_test) == 25000
        
        assert len(y_train) == 15000
        assert len(y_val) == 10000
        assert len(y_test) == 25000
    elif dataset_name.lower() == 'california_housing':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.california_housing.load_data(version="small")
        x_val = x_train[400:]
        y_val = y_train[400:]
        x_train = x_train[:400]
        y_train = y_train[:400]
        
        assert x_train.shape == (400, 8)
        assert x_val.shape == (80, 8)
        assert x_test.shape == (120, 8)
        
        assert y_train.shape == (400,)
        assert y_val.shape == (80,)
        assert y_test.shape == (120,)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if verbose:
            print("\n" + "="*40)
            print("DATASET SHAPES".center(40))
            print("="*40)
            print(f"{'Training Data (x):':<25}{x_train.shape}")
            print(f"{'Training Labels (y):':<25}{y_train.shape}")
            print("-" * 40)
            print(f"{'Validation Data (x):':<25}{x_val.shape}")
            print(f"{'Validation Labels (y):':<25}{y_val.shape}")
            print("-" * 40)
            print(f"{'Test Data (x):':<25}{x_test.shape}")
            print(f"{'Test Labels (y):':<25}{y_test.shape}")
            print("="*40 + "\n")
            
    return (x_train, y_train, x_val, y_val, x_test, y_test)