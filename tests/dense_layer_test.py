'''
What is this function obligated to do?
1) Construct an object to test
2) Call one thing
3) Assert that the obligation is fulfilled
'''
import pytest
import tensorflow as tf

from nn_lib.layers.core_layers import Dense

@pytest.fixture
def basic_layer():
    d1 = Dense("dense_prev", 10, "linear", prev_layer=None)
    return Dense("test_layer", 20, "relu", prev_layer=d1)

def test_getters(basic_layer):
    assert basic_layer.get_name() == "test_layer"
    assert basic_layer.get_act_func() == "relu"
    assert basic_layer.get_prev_layer()
    assert basic_layer.get_wts() is None
    assert basic_layer.get_b() is None
    assert basic_layer.has_wts()
    assert basic_layer.num_units

def test_set_mode(basic_layer):
    assert not basic_layer.get_mode().numpy()
    basic_layer.set_mode(True)
    assert basic_layer.get_mode().numpy()
    basic_layer.set_mode(False)
    assert not basic_layer.get_mode().numpy()

def test_lazy_init_params(basic_layer):
    input_shape = (16, 32, 32, 3)
    assert basic_layer.get_wts() is None
    assert basic_layer.get_b() is None
    basic_layer.init_params(input_shape=input_shape)
    assert basic_layer.get_wts().shape == (3, 20)
    assert basic_layer.get_b().shape == (20, )

def test_get_params(basic_layer):
    assert basic_layer.get_params() == []
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    basic_layer(x)
    params = basic_layer.get_params()
    assert len(params) == 2
    assert params[0].shape == (2, 20)
    assert params[1].shape == (20, )

@pytest.mark.parametrize("activation, expected", [
    ("relu", tf.math.sqrt(2.0)),
    ("linear", tf.math.sqrt(1.0)),
    ("softmax", tf.math.sqrt(1.0))
])
def test_kaiming_gain_valid(activation, expected):
    layer = Dense("l", units=20, activation=activation)
    result = layer.get_kaiming_gain()
    assert abs(float(result) - float(expected)) < 1e-6

def test_kaiming_gain_invalid():
    layer = Dense("l", units=20, activation="invalid_act")
    with pytest.raises(ValueError, match="not supported"):
        layer.get_kaiming_gain()

@pytest.mark.parametrize("activation", ["relu", "linear", "softmax"])
def test_compute_net_act_shape(activation):
    layer = Dense("l", 20, activation=activation)
    x = tf.constant([[-1.0, 0.0, 1.0]])
    result = layer.compute_net_act(x)
    assert result.shape == x.shape
    
def test_compute_net_act_invalid():
    layer = Dense("l", 20, activation="invalid_act")
    with pytest.raises(ValueError):
        layer.compute_net_act(tf.constant([[1.0]]))
    
