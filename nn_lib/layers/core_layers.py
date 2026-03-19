import tensorflow as tf

class Layer:
    '''Parent class for all network layers. Implements shared functionality common across all layers.'''

    def __init__(self, layer_name, activation, prev_layer, do_bn=False, bn_momentum=0.99, do_ln= False):
        self.layer_name = layer_name
        self.act_func = activation
        self.prev_layer = prev_layer
        self.is_training = tf.Variable(False, trainable=True,
                                       name=self.layer_name.replace("/", "_").replace(" ", "_") + "_is_training")

        # use lazy initialization on first call
        self.wts = None
        self.b = None
        self.output_shape = None

        # batch norm params
        self.do_bm = do_bn
        self.bm_momentum = bn_momentum
        self.bn_gain = None
        self.bn_bias = None
        self.bn_mean = None
        self.bn_std = None
        
        # layer norm
        self.do_ln = do_ln
        self.ln_gain = None
        self.ln_bias = None

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
        if self.bn_gain is not None:
            params.append(self.bn_gain)
        if self.bn_bias is not None:
            params.append(self.bn_bias)
        if self.ln_gain is not None:
            params.append(self.ln_gain)
        if self.ln_bias is not None:
            params.append(self.ln_bias)

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

    def is_doing_bn(self):
        return self.is_doing_bn
    
    def init_bn_params(self):
        # TODO
        pass
        
    def compute_bn(self, net_in, eps=0.001):
        pass

    def init_ln_params(self, x):
        # TODO
        pass

    def compute_ln(self, x, eps=0.001):
        # TODO
        pass
        
    def init_params(self, input_shape):
        pass

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
            raise ValueError(f"Unknown activation function {self.act_func}jk")

    def __call__(self, x):
        net_in = self.compute_net_in(x=x)
        net_act = self.compute_net_act(net_in=net_in)

        # store output shape on first call
        if self.output_shape is None:
            self.output_shape = list(net_act.shape)

class Dense(Layer):
    
    def __init__(self, name, units, activation="relu", wt_scale=1e3, prev_layer=None,
                 wt_init="normal", do_bn=False, do_ln=False):
        super().__init__(layer_name=name, activation=activation, prev_layer=prev_layer,
                         do_bn=do_bn, do_ln=do_ln)
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
        # TODO
        pass