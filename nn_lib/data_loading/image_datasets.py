'''datasets.py
Loads and preprocesses data for training/evaluating neural networks.
'''
# import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def norm_imgs(norm_method, x_train, x_test, eps=1e-10):
    if norm_method == 'global':
        rgb_mean = tf.math.reduce_mean(x_train, axis=[0,1,2])
        rgb_std = tf.math.reduce_std(x_train, axis=[0,1,2])

        x_train = (x_train - rgb_mean) / (rgb_std + eps)
        x_test = (x_test - rgb_mean) / (rgb_std + eps)

    elif norm_method == 'center':
        mean = tf.math.reduce_mean(x_train, axis=[0,1,2])

        x_train = x_train - mean
        x_test = x_test - mean

    elif norm_method == 'none':
        pass
    else:
        raise ValueError(f"Unknown normalization method {norm_method}. Supported options: 'global', 'center', 'none'")

    return x_train, x_test
    
def load_img_dataset(name, norm_method="global", flatten=True, verbose=False):
    # NOTE: datasets load as np arrays
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        classnames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        raise ValueError(f"Unknow dataset {name}. Supported options: 'mnist', 'cifar10'")

    # ensure in tf obj format and normalize btwn 0-1
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.0
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.0
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

    # flatten labels
    y_train = tf.reshape(y_train, [-1,])
    y_test = tf.reshape(y_test, [-1,])

    # add singleton dim so shapes are (N, Iy, Ix, D) for both datasets
    if len(x_train.shape) == 3:
        x_train = tf.expand_dims(x_train, axis=-1)
        x_test = tf.expand_dims(x_test, axis=-1)

    x_train, x_test = norm_imgs(norm_method, x_train, x_test)

    if flatten:
        x_train = tf.reshape(x_train, [x_train.shape[0], -1])
        x_test = tf.reshape(x_test, [x_test.shape[0], -1])

    if verbose:
        print(f"[DEBUG] Fetching '{name}' dataset using norm method '{norm_method}'. Flattening data? {flatten}")
        print(f"[DEBUG] x_train.shape: {x_train.shape} | type: {type(x_train)}")
        print(f"[DEBUG]     x_train min/max: {tf.reduce_min(x_train):.02f}/{tf.reduce_max(x_train):.02f}")
        print(f"[DEBUG] x_test.shape: {x_test.shape} | type: {type(x_test)}")
        print(f"[DEBUG]     x_test min/max: {tf.reduce_min(x_test):.02f}/{tf.reduce_max(x_test):.02f}")
        print(f"[DEBUG] y_train.shape: {y_train.shape} | type: {type(y_train)}")
        print(f"[DEBUG]     y_train min/max: {tf.reduce_min(y_train):.02f}/{tf.reduce_max(y_train):.02f}")
        print(f"[DEBUG] y_test.shape: {y_test.shape} | type: {type(y_test)}")
        print(f"[DEBUG]     y_test min/max: {tf.reduce_min(y_test):.02f}/{tf.reduce_max(y_test):.02f}")
    
    return x_train, y_train, x_test, y_test, classnames

def train_val_split(x_train, y_train, prop_val=0.1):
    '''Divides a training set into training & validation splits.'''
    num_train = int(x_train.shape[0] * (1 - prop_val))

    x_val = x_train[num_train:]
    y_val = y_train[num_train:]

    x_train = x_train[:num_train]
    y_train = y_train[:num_train]

    return x_train, y_train, x_val, y_val

def plot_imgs(x, y, classnames, plt_sz=7, reshape_dims=None):

    fig, axes = plt.subplots(plt_sz, plt_sz, figsize=(10,10))
    axes = axes.flatten()

    if reshape_dims:
        # add batch to reshape_dims
        reshape_dims.insert(0, x.shape[0])
        x = tf.reshape(x, reshape_dims)

    for i in range(plt_sz**2):
        img = x[i]
        label_idx = int(y[i])

        axes[i].imshow(img)
        axes[i].set_title(classnames[label_idx], fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()