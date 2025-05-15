'''core_layers.py

Conatains the general layer classes for building neural networks.

Maddie Puzon
May 2025

=== SHAPES REFERENCE ===

N : number of samples / number of tokens in corpus
B : mini-batch size
M : number of features / number of tokens in vocab
C : number of classes
K : number of filters (conv2d)
    K1: number of filters/units in previous layer
    K2: number of filters/units in current layer
H : number of hidden units (dense) / embedding dimention
R : dropout rate
D : number of color channels
T : length of sequence processed by a transformer
A : number of attention heads
Iy, Ix : image dimentions
Fy, Fx : filter dimentions (conv2d)

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
        
    def get_name(self) -> str:
        '''Returns the human-readable string name of this Layer.'''
        return self.layer_name
    
    def get_act_func_name(self) -> str:
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

    def get_mode(self) -> bool:
        '''Returns whether the Layer is in a training state.'''
        return self.is_training

    def set_mode(self, is_training) -> None:
        '''Informs the layer whether the neural network is currently training.

        Parameters:
        -----------
        is_training: bool.
            True if the network is currently training, False otherwise.
        '''
        # need to use .assign() not = operator!
        self.is_training.assign(is_training)
        
    def compute_net_acts(self, net_in) -> tf.constant:
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

    def __call__(self, x) -> tf.constant:
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
    
    def get_params(self) -> list:
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
        '''Returns if this Layer has weights.'''
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
    
    @abstractmethod
    def __str__(self):
        '''A nicely formatted string representing this Layer'''
        raise NotImplementedError

# --------------- CORE ML LAYERS --------------- #

class Dense(Layer):
    '''Neural network layer that uses dense net input.'''
    def __init__(self, name, units, prev_section, activation="relu", 
                 do_batch_norm=False, do_layer_norm=False) -> None:
        '''Dense layer constructor.
        
        Parameters:
        -----------
        name: str.
            Human-readable name for this Layer; used for printing and debugging.
        units: int.
            Number of units in the layer (H).
        activation: str.
            Name of activation function to apply within this Layer.
        prev_section: Layer.
            Reference to the Layer object beneath this Layer.
        do_batch_norm: bool.
            Whether to do batch normalization in this Layer.
        do_layer_norm: bool.
            Whether to do layer normalization in this Layer.
        '''
        super().__init__(layer_name=name,
                         activation=activation,
                         prev_section=prev_section,
                         do_batch_norm=do_batch_norm,
                         do_layer_norm=do_layer_norm)
        self.num_units = units
        
    def has_wts(self) -> bool:
        return True
    
    def init_params(self, input_shape) -> None:
        '''Initializes the Dense layer's weights and biases.
        
        Parameters:
        -----------
        input_shape: list.
            The shape of the input that the layer will process (B, M).
        '''
        B = input_shape[0]
        M = input_shape[-1]
        H = self.num_units
        
        bias_trainable = (not self.is_using_batch_norm()) or (not self.is_using_layer_norm)
        # always use He weight intialization
        stdev = self.get_kaiming_gain() / tf.math.sqrt(float(M))
        self.wts = tf.Variable(tf.random.normal(shape=(M, self.num_units), stddev=stdev),
                               name=self.layer_name.replace(" ", "_").replace("/", "_") + "_wts",
                               trainable=True)
        self.b = tf.Variable(tf.zeros(shape=(H)),
                             name=self.layer_name.replace(" ", "_").replace("/", "_") + "_bias",
                             trainable=(bias_trainable)) 
                             # only use bias if not doing batch/layer norm

    def compute_net_input(self, x):
        '''Calculates the net input for this Dense layer.
        
        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, M)
            Input from the previous layer in the network.
            
        Returns:
        --------
        net_in: tf.constant. tf.float32s. shape=(B, H)
            The net input for this Layer.
        '''
        # uses lazy initialization
        if self.wts is None:
            self.init_params(input_shape=x.shape)
        return x @ self.wts + self.b
    
    def compute_batch_norm(self, net_in, eps=0.001):
        '''Computes batch normalization for Dense layers.
        
        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, H)
            The net input computed on the current mini-batch.
        eps: float.
            A small number to prevent division by 0 when standardizing net_in.
            
        Returns:
        --------
        net_in_bn: tf.constant. tf.float32. shape=(B, H)
            The net_in standardized according to the batch norm algorithm.
        '''
        if not self.do_batch_norm:
            raise ValueError(f"self.do_batch_norm is False, for layer {self.layer_name}. Either should not be calling compute_batch_norm or do_batch_norm parameter is initialized incorrectly.")

        if self.is_training: # use current batch statistics if training net 
            # compute batch mean and standard deviation
            cur_batch_mean = tf.reduce_mean(input_tensor=net_in, axis=0, keepdims=True)
            cur_batch_stdev = tf.math.reduce_std(input_tensor=net_in, axis=0, keepdims=True)

            net_in_j = (net_in - cur_batch_mean) / (cur_batch_stdev + eps)
        else: # use moving averages if doing  prediction/testing
            net_in_j = (net_in - self.bn_mean) / (self.bn_stdev + eps)
        
        # update moving averages if in training mode
        if self.is_training: 
            self.bn_mean.assign(self.batch_norm_momentum * self.bn_mean + (1 - self.batch_norm_momentum) * cur_batch_mean)
            self.bn_stdev.assign(self.batch_norm_momentum * self.bn_stdev + (1 - self.batch_norm_momentum) * cur_batch_stdev)
            
        return self.bn_gain * net_in_j + self.bn_bias
            
    def __str__(self):
        '''This Layer's method to print a nicely formated representation of itself.'''
        return f"Dense layer '{self.layer_name}' | output_shape: {self.output_shape}"

class Dropout(Layer):
    '''Neural network layer which zeros out a proportion of the net_in signals; prevents overfitting.'''
    def __init__(self, name, rate, prev_section) -> None:
        '''Dropout layer constructor.
        
        Parameters:
        -----------
        name: str.
            Human-readable name for the this Layer; used for debugging and printing.
        rate: float.
            Proportion between 0.0 and 1.0 of net_in signals to drop within each mini-batch.
        prev_section: Layer.
            Reference to the Layer below this one.
        '''
        super().__init__(layer_name=name, 
                         activation="linear",
                         prev_section=prev_section)
        self.dropout_rate = rate
        
    def compute_net_input(self, x):
        '''Somputes net_in for this Dropout layer.
        
        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...)
            Input from the Layer below this one; can be of any dimention.
            
        Returns:
        --------
        net_in: tf.constant, tf.float32s. shape=(B, ...)
            The net input with proportion zeroed out.
        '''
        # pass all data through if not training
        if not self.is_training:
            net_in = x
        else: # if network is training
            mask = tf.random.uniform(tf.shape(x), dtype=x.dtype)
            # convert to binary mask
            mask = tf.where(mask >= self.dropout_rate, 1.0, 0.0)
            # apply mask
            net_in = x * mask
            # scale so total net_in values are similar in training and testing
            net_in = net_in * tf.cast(1.0 / (1.0 - self.dropout_rate), x.dtype)
        
        return net_in

    def has_wts(self):
        return False
    
    def init_params(self, input_shape):
        raise TypeError(f"Dropout layer does not have params.")
    
    def compute_batch_norm(self, net_in, eps=0.001):
        raise TypeError(f"Dropout layer cannot implement batch norm.")
    
    def __str__(self):
        '''This Layer's method to print a nicely formated representation of itself.'''
        return f"Dropout layer '{self.layer_name}' | output_shape: {self.output_shape}"
    
class Flatten(Layer):
    '''A neural network layer that flattens out all non-batch dimentions of its input.'''
    def __init__(self, name, prev_section):
        '''Flatten Layer constructor.
        
        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer.
        prev_section: Layer.
            Reference to the Layer object below this one.
        '''
        super().__init__(layer_name=name, activation="linear", prev_section=prev_section)

    def compute_net_input(self, x) -> tf.constant:
        '''Computes the net input for this Flatten layer.
        
        Parameters:
        -----------
        x. tf.constant. tf.float32s. shape=(B, ...)
            Input from the Layer below this one. Generally from a Conv2d block, so shape
            is usually (B, Iy, Ix, K).
            
        Returns:
        --------
        tf.constant. tf.float32s. shape(B, F)
            The net_in. `F` here is the product of all the dimentions of x after the batch dim.
        '''
        return tf.reshape(x, (tf.shape(x)[0], -1))

    def has_wts(self) -> bool:
        return False
    
    def init_params(self, input_shape):
        raise TypeError(f"Flatten layer does not have params.")
    
    def compute_batch_norm(self, net_in, eps=0.001):
        raise TypeError(f"Flatten layer cannot implement batch norm.")
    
    def __str__(self):
        '''This Layer's method to print a nicely formated representation of itself.'''
        return f"Flatten layer '{self.layer_name}' | output_shape: {self.output_shape}"
    
    
class MaxPool2D(Layer):
    '''A neural network layer which applies MaxPooling to its inputs.'''
    def __init__(self, name, prev_section, pool_size=(2,2), strides=1, padding='VALID'):
        '''Max pooling layer constructor.
        
        Parameters:
        -----------
        name: str.
            Human-readable string.
        pool_size. tuple. len(pool_size)=2.
             The horizontal and verticle size of the pooling window.
        strides. int.
            the horizontal and verticle stride of the maxpooling operation.
        prev_section: Layer.
            Reference to the Layer object that is beneath this one.
        padding: str.
            Whether or not to pad a single input signal before performing max pooling,
            supported options: "VALID", "SAME".
           '''
        super().__init__(layer_name=name, activation="linear", prev_section=prev_section)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        
    def compute_net_input(self, x) -> tf.constant:
        '''Computes the net input for the current Maxpool2D layer.
        
        Parameters:
        -----------
        x: tf.constatnt. tf.float32s. shape=(B, Iy, Ix, K1).
            Input from the layer beneath this one. Should be 4D; K1 refers to the
            number of units/filters in the previous lauer.
        
        Returns:
        --------
        net_in: tf.constant. tf.float32. shape=(B, Iy, Ix, K2).
            The net input; K2 refers to the number of units/filters in the current layer.
        '''
        return tf.nn.max_pool2d(input=x, ksize=self.pool_size, strides=self.strides, padding=self.padding)
                 
    def has_wts(self) -> bool:
        return False
    
    def init_params(self, input_shape):
        raise TypeError(f"MaxPool2D layer does not have params.")
    
    def compute_batch_norm(self, net_in, eps=0.001):
        raise TypeError(f"MaxPool2D layer cannot implement batch norm.")
    
    def __str__(self):
        '''This Layer's method to print a nicely formated representation of itself.'''
        return f"MaxPool2D layer '{self.layer_name}' | output_shape: {self.output_shape}"

class Conv2D(Layer):
    '''A neural network that performs a 2D convolution operation on the given input.'''
    def __init__(self, name, units, prev_section, kernel_size=(1,1), strides=1, activation="relu", 
                 do_batch_norm=False, do_layer_norm=False):
        '''Conv2D layer constructor.'''
        super().__init__(layer_name=name, activation=activation, prev_section=prev_section,
                         do_batch_norm=do_batch_norm, do_layer_norm=do_layer_norm)
        self.num_units = units
        self.kernel_size = kernel_size
        self.strides = strides

    def has_wts(self) -> bool:
        return True
    
    def init_params(self, input_shape) -> None:
        '''Initializes the Conv2D layer's weights and biases.
        
        Parameters:
        -----------
        input_shape: list. len(input_shape)=4.
            The shape of mini-batched of input passed to this Layer, shape=(B, Iy, Ix, K1).
            K1 is the number of units/filters in the previous layer.
        '''
        Fy, Fx = self.kernel_size
        K1 = input_shape[-1]
        
        # always use He wt initialization
        std = self.get_kaiming_gain() / tf.math.sqrt(float(K1*Fy*Fx))
        self.wts = tf.Variable(tf.random.normal(shape=(Fy, Fx, K1, self.num_units), stddev=std),
                                name=self.layer_name.replace(" ", "_").replace("/", "_") + "_wts",
                                trainable=True)
        self.b = tf.Variable(tf.zeros(shape=(self.num_units)),
                                name=self.layer_name.replace(" ", "_").replace("/", "_") + "_bias",
                                trainable=(not self.is_doing_batchnorm()))
    
    def compute_net_input(self, x):
        '''Computes the net input for this Conv2D layer using SAME boundary conditions.
        
        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            Input from Layer beneath this one.
            
        Returns:
        --------
        net_in: tf.constant. tf.float32s. shape=(B, Iy, Ix, K2).
            The net input for this Conv2D layer.
        '''
        # use lazy initialization
        if self.wts is None:
            self.init_params(input_shape=x.shape)
        
        return tf.nn.conv2d(input=x, filters=self.wts, strides=self.strides, padding="SAME") + self.b
    
    def compute_batch_norm(self, net_in, eps=0.001):
        '''Computes the batch normalization for a Conv2D layer.
        
        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, Iy, Ix, K).
            The net input computed on the current mini-batch.
        eps: float.
            A small value to prevent division by 0 when standardizing net_in.
            
        Returns:
        --------
        net_in_bn: tf.constant. tf.float32. shape=(B, Iy, Ix, K).
            The net_in, standardized according to the batch norm algorithm.
        '''
        if not self.do_batch_norm:
            raise ValueError(f"self.do_batch_norm is False for layer {self.layer_name}. Either should not be calling compute_batch_norm or do_batch_norm parameter is initialized incorrectly.")
        
        # compute batch statistics
        cur_batch_mean = tf.reduce_mean(net_in, axis=[0,1,2], keepdims=True)
        cur_batch_stddev = tf.math.reduce_std(net_in, axis=[0,1,2], keepdims=True)
        
        # compute netIns for all neurons in layer
        if self.is_training:
            net_in_j = (net_in - cur_batch_mean) / (cur_batch_stddev + eps)
        else: # testing/prediction
            net_in_j = (net_in - self.bn_mean) / (self.bn_stdev + eps)
            
        # update moving averages if training
        if self.is_training:
            self.bn_mean.assign(self.batch_norm_momentum * self.bn_mean + (1 - self.batch_norm_momentum) * cur_batch_mean)
            self.bn_stdev.assign(self.batch_norm_momentum * self.bn_stdev + (1 - self.batch_norm_momentum) * cur_batch_stddev)
            
        # apply learned shift and scaling
        net_in_bn = self.bn_gain * net_in_j + self.bn_bias
        return net_in_bn
    
    def __str__(self):
        '''This Layer's method to print a nicely formated representation of itself.'''
        return f"Conv2D layer '{self.layer_name}' | output_shape: {self.output_shape}"