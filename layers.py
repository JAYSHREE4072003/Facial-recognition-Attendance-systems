"""
Custom layers for the Siamese Network.

Keeping this in its own module matters: when you reload a saved .h5 model
with tf.keras.models.load_model(), Keras needs to know how to reconstruct
any custom layer classes. Importing L1Dist from here (instead of redefining
it inline in a notebook) avoids the "unknown layer" errors that show up when
the class lives only in __main__.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class L1Dist(Layer):
    """
    Siamese L1 (absolute difference) distance layer.

    Given two embedding vectors (anchor/input and validation), returns the
    element-wise absolute difference between them. This is fed into a final
    Dense(1, sigmoid) layer that learns to turn "distance" into a
    same/different probability.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Accepts a [input_embedding, validation_embedding] list/tuple.

        Note: this takes a single `inputs` argument (a pair of tensors)
        rather than two separate positional arguments. Older Keras/TF
        versions (e.g. TF 2.4, as used in the original project) tolerated
        `call(self, a, b)` called as `layer(a, b)`, but modern Keras 3
        expects a single input, so the layer must be invoked as
        `layer([a, b])`. This version works on both.
        """
        # Keras 3's functional-API tracing can wrap each element of the pair
        # in its own extra list (e.g. [[emb_a], [emb_b]]) depending on how
        # the layer is invoked, so unwrap defensively rather than assuming
        # a fixed nesting depth.
        flat = tf.nest.flatten(inputs)
        input_embedding, validation_embedding = flat[0], flat[1]
        return tf.math.abs(input_embedding - validation_embedding)
