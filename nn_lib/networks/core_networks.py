import time
import tensorflow as tf
from numpy import random

class DeepNet:
    
    def __init__(self, input_feats_shape, reg=0):
        self.input_feats_shape=input_feats_shape
        self.reg = reg
        self.opt = None
        
        self.opt_name = None
        self.loss_name = None
        self.output_layer = None
        self.layers = None
        self.all_net_params = []
        
        self.train_loss_hist = None
        self.val_loss_hist = None
        self.val_acc_hist = None
        self.iter_trained = None


    def compile(self, loss="cross_entropy", optimizer="adam", lr=1e-3, beta=0.9, summary=True):
        self.loss_name = loss
        self.opt_name = optimizer.lower()

        # init optimizer
        if optimizer == "adam":
            self.opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta)
        elif optimizer == "adamw":
            self.opt = tf.keras.optimizers.AdamW(learning_rate=lr, beta_1=beta)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Valid options: 'adam', 'adamw'")

        # intialize network w/ a fake forward pass
        x_fake = self.get_fake_input()
        self(x_fake)

        self.init_bn_params()
        
        if summary:
            self.summary()
            
        self.all_net_params = self.get_all_params()

    def get_fake_input(self):
        '''Generates a fake mini-batch of data to init wts.'''
        return tf.zeros(shape=(1, *self.input_feats_shape))

    def get_all_params(self, wts_only=False):
        all_net_params = []

        layer = self.output_layer
        while layer:
            if wts_only:
                params = layer.get_wts()
                if not params:
                    params = []
                if not isinstance(params, list):
                    params = [params]
            else:
                params = layer.get_params()

            all_net_params.extend(params)
            layer = layer.get_prev_layer()

        return all_net_params

    def set_layer_training_mode(self, is_training):
        layer = self.output_layer
        while layer:
            layer.set_mode(is_training)
            layer = layer.get_prev_layer()
            
    def init_bn_params(self):
        # TODO
        pass
    
    def update_params(self, tape, loss):
        grads = tape.gradient(loss, self.all_net_params)
        self.opt.apply_gradients(zip(grads, self.all_net_params))

    def accuracy(self, y_true, y_pred):
        corr_preds = float(tf.math.count_nonzero(
            tf.math.equal(tf.cast(y_true, tf.int16), tf.cast(y_pred, tf.int16)))
        )
        num_samps = float(y_true.shape[0])

        return corr_preds / num_samps
    
    def predict(self, x, output_net_act=None):
        if not output_net_act:
            output_net_act = self(x)
        
        return tf.argmax(output_net_act, axis=1)

    def loss(self, out_net_act, y, eps=1e-16):
        batch_loss = None
        if self.loss_name == "cross_entropy":
           scc = tf.keras.losses.SparseCategoricalCrossentropy() 
           batch_loss = scc(y, out_net_act)
        elif self.loss == "mae":
            mae = tf.keras.losses.MeanAbsoluteError()
            batch_loss = mae(y, out_net_act)
        elif self.loss_name == "mse":
            mae = tf.keras.losses.MeanSquaredError()
            batch_loss = mae(y, out_net_act)
        else:
            raise ValueError(f"Unknow loss function '{self.loss_name}'. Valid options: 'cross_entropy', 'mae', 'mse'")

        # adam regularization
        if self.opt_name == 'adam':
            all_net_wts = self.get_all_params(wts_only=True)
            reg_term = self.reg * 0.5 * tf.reduce_sum(
                [tf.reduce_sum(wts**2) for wts in all_net_wts]
            )
            batch_loss += reg_term

        return batch_loss
        
    def summary(self):
        print(50*'-')
        layer = self.output_layer
        while layer:
            print(layer)
            layer = layer.get_prev_layer()
        print(50*'-')

    def evaluate(self, x, y, batch_sz=64):
        self.set_layer_training_mode(False)

        N = len(x)
        if batch_sz > N:
            batch_sz = N
        if batch_sz < 1:
            raise ValueError(f"Batch size must be positive & non-zero; batch_sz={batch_sz}")
            
        num_batches = N // batch_sz
        loss, acc = 0, 0
        for b in range(num_batches):
            curr_x = x[b * batch_sz: (b+1) * batch_sz]
            curr_y = y[b * batch_sz: (b+1) * batch_sz]
        
            curr_acc, curr_loss = self.test_step(x_batch=curr_x, y_batch=curr_y)
            acc += curr_acc
            loss += curr_loss
            
        acc /= num_batches
        loss /= num_batches

        return acc, loss

    # @tf.function(jit_compile=True)
    def train_step(self, x_batch, y_batch):
        cur_loss = None
        with tf.GradientTape() as tape:
            output_net_act = self(x_batch)
            cur_loss = self.loss(output_net_act, y_batch)
            self.update_params(tape, cur_loss)

        return cur_loss

    # @tf.function(jit_compile=True)
    def test_step(self, x_batch, y_batch):
        net_acts = self(x=x_batch)
        y_pred = self.predict(x=x_batch, output_net_act=net_acts)
        loss = self.loss(out_net_act=net_acts, y=y_batch)
        acc = self.accuracy(y_true=y_batch, y_pred=y_pred)

        return acc, loss

    def fit(self, x, y, x_val=None, y_val=None, batch_size=128, max_epochs=9999, val_every=1, verbose=True,
             patience=None, lr_patience=999, lr_decay_factor=0.5, lr_max_decays=10):

        self.set_layer_training_mode(True)
        N = x.shape[0]
        
        BATCHES_PER_EPOCH = int(N / batch_size)

        train_loss_hist = []
        val_acc_hist = []
        val_loss_hist = []
        patience_tracker = []
        lr_tracker = [] # for lr decay
        num_decays = 0

        rng = random.default_rng(0)
        for cur_epoch in range(max_epochs):
            batches = rng.integers(0, N, size=(BATCHES_PER_EPOCH, batch_size))
            
            st = time.time()
            epoch_loss_hist = []
            for cur_batch in range(BATCHES_PER_EPOCH):
                mini_b = tf.gather(x, batches[cur_batch], axis=0)
                mini_y = tf.gather(y, batches[cur_batch], axis=0)

                mini_loss = self.train_step(x_batch=mini_b, y_batch=mini_y)
                epoch_loss_hist.append(mini_loss)

            epoch_time = time.time() - st
            time_proj = (max_epochs - cur_epoch) * epoch_time
            if verbose:
                print(f"Beginning Epoch: {cur_epoch + 1}/{max_epochs} | Time(taken): {epoch_time:.2f} | Time(proj): {time_proj:.2f}")
            
            epoch_train_loss = float(tf.reduce_mean(epoch_loss_hist))
            train_loss_hist.append(epoch_train_loss)

            # check validation accuracy
            if cur_epoch % val_every == 0:
                val_acc, val_loss = self.evaluate(x=x_val, y=y_val, batch_sz=batch_size)
                self.set_layer_training_mode(False)
                val_acc_hist.append(float(val_acc))
                val_loss_hist.append(float(val_loss))

            # check for early stopping or lr decrease
            if patience and (cur_epoch % val_every == 0):
                # early stopping
                patience_tracker, stop = self.early_stopping(patience_tracker, val_loss_hist[-1], patience)
                # lr_decay
                lr_tracker, lr_stop = self.early_stopping(lr_tracker, val_loss_hist[-1], lr_patience)

                if lr_stop and (num_decays < lr_max_decays):
                    self.update_lr(decay_factor=lr_decay_factor)
                    num_decays += 1
                    lr_tracker = [] # reset lr tracker after decay

                if stop:
                    if verbose:
                        print(f"Finished training early after {cur_epoch}/{max_epochs} epochs!")
                    self.set_layer_training_mode(False)
                    self.train_loss_hist = train_loss_hist
                    self.val_loss_hist = val_loss_hist
                    self.val_acc_hist = val_acc_hist
                    self.iter_trained = cur_epoch
                    return train_loss_hist, val_loss_hist, val_acc_hist, cur_epoch 
            if verbose:
                print(f"Completed epoch: {cur_epoch+1} | Train loss: {train_loss_hist[-1]:.6f} | Val loss: {val_loss_hist[-1]:.6f} | Val acc: {val_acc_hist[-1]:.6f}")
        
        if verbose:
            print(f"Finished training after {cur_epoch+1}/{max_epochs} epochs!")

        self.set_layer_training_mode(False)
        self.train_loss_hist = train_loss_hist
        self.val_loss_hist = val_loss_hist
        self.val_acc_hist = val_acc_hist
        self.iter_trained = cur_epoch
        return train_loss_hist, val_loss_hist, val_acc_hist, cur_epoch 

    def __call__(self, x):
        net_act = x
        for layer in self.layers:
            net_act = layer(net_act)
        return net_act