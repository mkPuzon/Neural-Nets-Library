'''core_layer.py

Standard layers for basic MLPs and CNNS.

Apr 2026
'''
import tensorflow as tf

class Layer:
    '''Parent class for all network layers. Implements shared functionality common across all layers.'''

    def __init__(self, layer_name, activation, prev_layer=None):
        self.layer_name = layer_name
        self.act_func = activation
        self.prev_layer = prev_layer
        self.is_training = tf.Variable(False, trainable=False, name=self.layer_name + "_is_training")

        # use lazy initialization to determine shapes on first call
        self.wts = None
        self.b = None
        self.output_shape = None

    def get_name(self):
        return self.layer_name
    
    def get_act_func(self):
        return self.act_func
    
    def get_prev_layer(self):
        return self.prev_layer

    def get_wts(self):
        return self.wts

    def get_b(self):
        return self.b

    def get_mode(self):
        return self.is_training

    def get_params(self):
        params = []
        
        if self.wts is not None:
            params.append(self.wts)
        if self.b is not None and self.b.trainable:
            params.append(self.b)

        return params

    def get_kaiming_gain(self):
        if self.act_func == "relu":
            return tf.cast(tf.math.sqrt(2.0), dtype=tf.float32)
        elif self.act_func == "linear":
            return tf.cast(tf.math.sqrt(1.0), dtype=tf.float32)
        elif self.act_func == "softmax":
            return tf.cast(tf.math.sqrt(1.0), dtype=tf.float32)
        else:
            raise ValueError(f"Kaiming wt init not supported for '{self.act_func}' activation")
        
    def set_mode(self, is_training):
        self.is_training.assign(is_training)

    def has_wts(self):
        return False

    def compute_net_in(self, x):
        pass

    def compute_net_act(self, net_in):
        if self.act_func == "relu":
            return tf.nn.relu(net_in)
        elif self.act_func == "linear":
            return net_in
        elif self.act_func == "softmax":
            return tf.nn.softmax(net_in)
        else:
            raise ValueError(f"Unknown activation function {self.act_func}")

    def __call__(self, x):
        net_in = self.compute_net_in(x=x)
        net_act = self.compute_net_act(net_in=net_in)

        # store output shape on first call
        if self.output_shape is None:
            self.output_shape = list(net_act.shape)
        
        return net_act

    def __repr__(self):
        return f"{self.layer_name}"

class Dense(Layer):
    
    def __init__(self, name, units, activation="relu", wt_scale=1e3, prev_layer=None, wt_init="normal"):
        super().__init__(layer_name=name, activation=activation, prev_layer=prev_layer)
        self.num_units = units
        self.wt_scale = wt_scale
        self.wt_init = wt_init.lower()
        
    def has_wts(self):
        return True 

    def init_params(self, input_shape):
        M = input_shape[-1]
        
        if self.wt_init == "normal":
            self.wts = tf.Variable(tf.random.normal(shape=(input_shape[-1], self.num_units), stddev=self.wt_scale))
            self.b = tf.Variable(tf.random.normal(shape=(self.num_units,), stddev=self.wt_scale))
        elif self.wt_init == "he":
            std = self.get_kaiming_gain() / tf.math.sqrt(float(M)) 
            self.wts = tf.Variable(tf.random.normal(shape=(M, self.num_units), stddev=std))
            self.b = tf.Variable(tf.zeros(shape=(self.num_units)))
        else:
            raise ValueError(f"Cannot initialize parameters with wt_init method {self.wt_init}")

    def compute_net_in(self, x):
        if self.wts is None:
            self.init_params(input_shape=x.shape)
        return x @ self.wts + self.b

    def compute_bn(self, net_in, eps=0.001):
        if not self.do_bn:
            raise ValueError(f"self.do_bn is False, for layer {self.layer_name}. Either should not be calling compute_bn or do_bn parameter is initialized incorrectly.")

        # compute batch statistics
        cur_batch_mean = tf.reduce_mean(net_in, axis=0, keepdims=True)
        cur_batch_std = tf.math.reduce_std(net_in, axis=0, keepdims=True)

        # compute net_in for all neurons in layer
        if self.is_training:
            net_in_j = (net_in - cur_batch_mean) / (cur_batch_std + eps)
        else:
            net_in_j = (net_in - self.bn_mean) / (self.bn_std + eps)

        # update moving averages if training
        if self.is_training:
            self.bn_mean.assign(self.bn_momentum * self.bn_mean + (1 - self.bn_momentum) * cur_batch_mean)
            self.bn_std.assign(self.bn_momentum * self.bn_std + (1 - self.bn_momentum) * cur_batch_std)

        # apply shift and scaling
        net_in_bn = self.bn_gain * net_in_j + self.bn_bias

        return net_in_bn

    def __str__(self):
        return f"Dense layer ({self.layer_name}) shape: {self.output_shape}"

    def __repr__(self):
        return f"Dense(name={self.layer_name}, activation={self.act_func}, wt_scale={self.wt_scale}, prev_layer={self.prev_layer}, \
        wt_init={self.wt_init}, do_bn={self.do_bn}, do_gn={self.do_gn})"
