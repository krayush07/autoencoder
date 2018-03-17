import tensorflow as tf


class Utils:
    def dense_layer(self, input, output_dim, name, activation=None):
        with tf.variable_scope(name):
            return tf.layers.dense(input, output_dim, activation=activation)

    def ffn_autoencoder(self, input, output_shape):
        layer3_op = self.ffn_encoder(input)
        layer6_op = self.ffn_decoder(layer3_op, output_shape)
        return layer6_op, layer3_op

    def ffn_decoder(self, layer3_op, output_shape):
        with tf.variable_scope('ffn_ae_dec'):
            layer4_op = self.dense_layer(layer3_op, 128, 'layer4', tf.nn.tanh)
            layer5_op = self.dense_layer(layer4_op, 256, 'layer5', tf.nn.tanh)
        with tf.variable_scope('ffn_ae_op'):
            layer6_op = self.dense_layer(layer5_op, output_shape, 'layer6')
        return layer6_op

    def ffn_encoder(self, input):
        with tf.variable_scope('ffn_ae_enc'):
            layer1_op = self.dense_layer(input, 256, 'layer1', tf.nn.tanh)
            layer2_op = self.dense_layer(layer1_op, 128, 'layer2', tf.nn.tanh)
        with tf.variable_scope('ffn_ae_rep'):
            layer3_op = self.dense_layer(layer2_op, 2, 'layer3')
        return layer3_op
