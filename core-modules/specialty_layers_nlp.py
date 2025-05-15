'''specialty_layers_nlp.py

Conatains specialized layer classes for building neural networks specializing in processing text data.

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
Iy, Ix : image dimentions
Fy, Fx : filter dimentions (conv2d)

========================

'''
import tensorflow as tf
import core_layers

class DenseEmbedding(core_layers.Dense):
    pass

class Embedding(core_layers.Layer):
    pass

class PositionalEncoding(core_layers.Layer):
    pass
