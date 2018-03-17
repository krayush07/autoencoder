import tensorflow as tf
from global_module.implementation_module import utils


class Autoencoder:
    def __init__(self, params):
        self.params = params
        self.create_placeholder()

    def create_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, None], name='input_placeholder')

    def autoencode(self):
        self.decoded_op, self.rep = utils.ffn_autoencoder(input, self.params.output_shape)

    def encode(self):
        self.rep = utils.ffn_encoder(self.input)

    def compute_loss(self):
        self.loss = tf.squared_difference(self.decoded_op, self.input)

    def train(self):
        global optimizer
        with tf.variable_scope('optimize_tar_net'):
            learning_rate = self.params.lr

            trainable_tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, trainable_tvars)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.params.max_grad_norm)
            grad_var_pairs = zip(grads, trainable_tvars)

            if self.params.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='sgd')
            elif self.params.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam')
            elif self.params.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=1e-6, name='adadelta')

            train_op = optimizer.apply_gradients(grad_var_pairs, name='apply_grad')

            return train_op
