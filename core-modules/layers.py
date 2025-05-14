'''layers.py

Conatains the general layer classes for building neural networks.

Maddie Puzon
May 2025

=== SHAPES REFERENCE ===

N : number of samples / number of tokens in corpus
B : mini-batch size
M : number of features / number of tokens in vocab
C : number of classes
k : number of filters (conv2d)
H : number of hidden units (dense) / embedding dimention
R : dropout rate
D : number of color channels
T : length of sequence processed by a transformer
A : number of attention heads
I_y, I_x : image dimentions
F_y, F_x : filter dimentions (conv2d)

========================

'''
import tensorflow as tf
from abc import ABC, abstractmethod

# --------------- ABSTRACT CLASS --------------- #

class Layer(ABC):
    '''Abstract parent class for all neural network layers.'''
    
    def __init__(self, layer_name, activation, prev_section, do_batch_norm=False,
                 batch_norm_momentum=0.99, do_layer_norm=False):
        self.layer_name = layer_name
        self.activation_name = activation
        self.prev_section = prev_section
        self.do_batch_norm = do_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.do_layer_norm = do_layer_norm
        
        self.wts = None
        self.b = None
        self.output_shape = None # list
        
        # tf.Variable b/c this boolean gets added to the static graph when compling static
        # graph with @tf.function decorator
        self.is_training = tf.Variable(False, trainable=False,
                                       name=self.layer_name.replace(" ", "_").replace("/", "_") + "_is_training")
        
        # batch norm params
        self.bn_gain = None
        self.bn_bias = None
        self.bn_mean = None
        self.bn_stdev = None

        # layer norm params
        self.ln_gain = None
        self.ln_bias = None
        
    def get_name(self):
        '''Returns the human-readable string name of this Layer.'''
        return self.layer_name
    
    def get_act_func_name(self):
        '''Returns the string name of the activation function used in this Layer.'''
        return self.activation_name
    
    def get_prev_section(self):
        '''Returns a reference to the layer or block below this one.'''
        return self.prev_section
    
    def get_wts(self):
        '''Returns the weights of the current layer'''
        return self.wts

    def get_b(self):
        '''Returns the bias of the current layer'''
        return self.b

    def get_mode(self):
        '''Returns whether the Layer is in a training state.'''
        return self.is_training

    def set_mode(self, is_training):
        '''Informs the layer whether the neural network is currently training.

        Parameters:
        -----------
        is_training: bool.
            True if the network is currently training, False otherwise.
        '''
        # need to use .assign() not = operator!
        self.is_training.assign(is_training)
        
    def compute_net_acts(self, net_in):
        '''Computes the activation for this Layer baed on the given net_in values.
        
        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, ...)
            The net input computed in this layer.
            
        Returns:
        --------
        net_act: tf.constant. tf.float32s. shape=(B, ...)
        '''
        if self.activation_name == "relu":
            return tf.nn.relu(net_in)
        elif self.activation_name == "linear" or self.activation_name == "identity":
            return net_in
        elif self.activation_name == "softmax":
            return tf.nn.softmax(net_in)
        elif self.activation_name == "gelu": # TODO: implement paper version
            return tf.nn.gelu(net_in)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")

    def __call__(self, x):
        '''Does a forward pass through this Layer with the given mini-batch, x.
        
        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...)
            The mini-batch of input data.
            
        Returns:
        --------
        net_act: tf.constant. tf.float32s. shape=(B, ...)
            The activation computed on the current mini-batch.
        '''
        if self.do_batch_norm and self.do_layer_norm:
            raise ValueError(f"Network should not use both batch norm and layer norm.")
        
        # if doing layer norm
        if self.do_layer_norm:
            if self.ln_gains is None or self.ln_biases is None:
                self.init_layer_norm_params(x)
            x = self.compute_layer_norm(x)
            
        # get net_in values
        net_in = self.compute_net_input(x=x)
            
        # if doing batch norm
        if self.do_batch_norm:
            if self.bn_gain is None or self.bn_gain is None:
                self.init_batch_norm_params()
            net_in = self.compute_batch_norm(net_in=net_in)
        
        # get net_act values
        net_act = self.compute_net_acts(net_in=net_in)
        
        # if we are processing first mini-batch, store shape of output
        if self.output_shape is None:
            self.output_shape = list(net_act.shape)
            
        return net_act
    
    def get_params(self):
        '''Returns a list of all the parameters learned by this Layer.'''
        params = []
        
        if self.wts is not None:
            params.append(self.wts)
        if self.b is not None and self.b.trainable:
            params.append(self.b)
        if self.bn_gain is not None:
            params.append(self.bn_gain)
        if self.bn_bias is not None:
            params.append(self.bn_bias)
        if self.ln_gain is not None:
            params.append(self.ln_gain)
        if self.ln_bias is not None:
            params.append(self.ln_bias)
    
    def get_kaiming_gain(self) -> float:
        '''Returns the appropriate Kaiming gain (float) for this Layer.'''
        if self.act_fun_name == "linear":
            return tf.cast(tf.math.sqrt(1.0), dtype=tf.float32)
        elif self.act_fun_name == "relu":
            return tf.cast(tf.math.sqrt(2.0), dtype=tf.float32)
        elif self.act_fun_name == "softmax":
            return tf.cast(tf.math.sqrt(1.0), dtype=tf.float32)
        elif self.act_fun_name == "gelu": # TODO: double check this value
            return tf.cast(tf.math.sqrt(1.0), dtype=tf.float32)
        else:
            raise ValueError(f'Kaiming wt initialization not supported for "{self.act_fun_name}" activation function.')
        
    def is_using_batch_norm(self) -> bool:
        '''Returns whether this Layer is using batch normalization.'''
        return self.do_batch_norm
    
    def init_batch_norm_params(self) -> None:
        '''Initializes the trainable and non-trainable parameters ued in batch normalization: bn_gain, bn_bias,
        bn_mean, and bn_stdev.'''
        # check if using batch norm
        if not self.do_batch_norm:
            return
        
        bn_params_shape = [1] * (len(self.output_shape) - 1)
        bn_params_shape.append(self.output_shape[-1])
        
        # shape=(k,) for dense, shape=(H,) for conv
        self.bn_gain = tf.Variable(tf.ones(bn_params_shape), trainable=True, 
                                   name=self.layer_name.replace(" ", "_").replace("/", "_") + "_bngain")
        self.bn_bias = tf.Variable(tf.zeros(bn_params_shape), trainable=True, 
                                   name=self.layer_name.replace(" ", "_").replace("/", "_") + "_bnbias")
        # moving averages--not trainable
        self.bn_mean = tf.Variable(tf.zeros(bn_params_shape), trainable=False, 
                                   name=self.layer_name.replace(" ", "_").replace("/", "_") + "_bnmean")
        self.bn_stdev = tf.Variable(tf.ones(bn_params_shape), trainable=False,
                                    name=self.layer_name.replace(" ", "_").replace("/", "_") + "_bnstdev")
    
        # disable standard bias neurons
        self.b = tf.Variable(0.0, trainable=False, 
                            name=self.layer_name.replace(" ", "_").replace("/", "_") + "_b_unused")
        
    def is_using_layer_norm(self) -> bool:
        '''Returns whether thsi layer is using layer normalization.'''
        return self.do_layer_norm
    
    def init_layer_norm_params(self, x) -> None:
        '''Initializes the parameters for layer normalization.
        
        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...)
            Input tensor to be normalized.
        '''
        # check if using layer norm
        if not self.do_layer_norm:
            return
        
        ln_params_shape = x.shape[1:]
        
        # shape=(k,) for dense, shape=(H,) for conv
        self.ln_gain = tf.Variable(tf.ones(ln_params_shape), trainable=True, 
                                   name=self.layer_name.replace(" ", "_").replace("/", "_") + "_lngain")
        self.ln_bias = tf.Variable(tf.zeros(ln_params_shape), trainable=True, 
                                   name=self.layer_name.replace(" ", "_").replace("/", "_") + "_lnbias")
    
        # disable standard bias neurons
        self.b = tf.Variable(0.0, trainable=False, 
                            name=self.layer_name.replace(" ", "_").replace("/", "_") + "_b_unused")
        
    def compute_layer_norm(self, x, eps=0.001):
        '''Computes layer normalization for the given input tensor, x. Normalizes the activations across neurons 
        in a layer instead of across a batch.
        
        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, M)
            Input tensor to be normalized.
        esp: float.
            A small constant added to stdev calculation to prevent division by 0.
            
        Returns:
        --------
        normalized_input: tf.constant. tf.float32s. shape=(B, M)
            The normalized tensor.
        '''
        layer_avg = tf.reduce_mean(input_tensor=x, axis=-1, keepdims=True)
        layer_std = tf.math.reduce_std(input_tensor=x, axis=-1, keepdims=True)
        
        x_norm = (x - layer_avg) / (layer_std + eps)
        
        return self.ln_gain * x_norm + self.ln_bias
    
    @abstractmethod
    def has_wts(self):
        '''Does the current layer store weights? By default, we assume it does not (i.e. always return False).'''
        raise NotImplementedError
    
    @abstractmethod
    def init_params(self, input_shape: tuple[int,...]) -> None:
        '''Initializes the Layer's parameters if it contains any.'''
        raise NotImplementedError
    
    @abstractmethod
    def compute_net_input(self, x):
        '''Computes net_in on the given input tensor, x.'''
        raise NotImplementedError
    
    @abstractmethod
    def compute_batch_norm(self, net_in, eps=0.001):
        '''Computes the batch norm based on the given net_in.'''
        raise NotImplementedError
    

# --------------- CORE ML LAYERS --------------- #

# class Dense(Layer):
    

# class Dropout(Layer):
    
    
# class Flatten(Layer):
    
    
# class MaxPool2D(Layer):
    
    
# class Conv2D(Layer):