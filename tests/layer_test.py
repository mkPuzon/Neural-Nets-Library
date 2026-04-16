'''
What is this function obligated to do?
1) Construct an object to test
2) Call one thing
3) Assert that the obligation is fulfilled
'''
import pytest
import tensorflow as tf

from nn_lib.layers.core_layers import Layer


@pytest.fixture
def basic_layer():
    return Layer("test_layer", "relu", prev_layer=None)

def test_getters(basic_layer):
    assert basic_layer.get_name() == "test_layer"
    assert basic_layer.get_act_func() == "relu"
    assert basic_layer.get_prev_layer() is None
    assert basic_layer.get_wts() is None
    assert basic_layer.get_b() is None
    assert not basic_layer.has_wts()

def test_set_mode(basic_layer):
    assert not basic_layer.get_mode().numpy()
    basic_layer.set_mode(True)
    assert basic_layer.get_mode().numpy()
    basic_layer.set_mode(False)
    assert not basic_layer.get_mode().numpy()

def test_get_params(basic_layer):
    params = basic_layer.get_params()
    assert params == []

@pytest.mark.parametrize("activation, expected", [
    ("relu", tf.math.sqrt(2.0)),
    ("linear", tf.math.sqrt(1.0)),
    ("softmax", tf.math.sqrt(1.0))
])
def test_kaiming_gain_valid(activation, expected):
    layer = Layer("l", activation=activation)
    result = layer.get_kaiming_gain()
    assert abs(float(result) - float(expected)) < 1e-6

def test_kaiming_gain_invalid():
    layer = Layer("l", activation="invalid_act")
    with pytest.raises(ValueError, match="not supported"):
        layer.get_kaiming_gain()

@pytest.mark.parametrize("activation", ["relu", "linear", "softmax"])
def test_compute_net_act_shape(activation):
    layer = Layer("l", activation=activation)
    x = tf.constant([[-1.0, 0.0, 1.0]])
    result = layer.compute_net_act(x)
    assert result.shape == x.shape
    
def test_compute_net_act_relu_clamps_negatives():
    layer = Layer("l", activation="relu")
    x = tf.constant([[-2.0, 0.0, 3.0]]) 
    result = layer.compute_net_act(x)
    assert float(result[0][0]) == 0.0
    assert float(result[0][2]) == 3.0

def test_compute_net_act_invalid():
    layer = Layer("l", activation="invalid_act")
    with pytest.raises(ValueError):
        layer.compute_net_act(tf.constant([[1.0]]))
    