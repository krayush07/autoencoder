import tensorflow as tf
from global_module.implementation_module import utils


class Autoencoder:
    def __init__(self, params, dir_obj):
        self.params = params
        self.model_utils = utils()
        self.dir_obj = dir_obj
        self.create_placeholder()
        self.autoencode()
        self.train_op = self.train()
        self.generate_summary()

    def create_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.params.output_shape], name='input_placeholder')

    def autoencode(self):
        self.decoded_op, self.rep = self.model_utils.ffn_autoencoder(self.input, self.params.output_shape)

    def encode(self):
        self.rep = self.model_utils.ffn_encoder(self.input)

    def compute_loss(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.decoded_op, self.input))

    def generate_summary(self):
        if self.params.mode == 'TR':
            train_loss = tf.summary.scalar('train_loss', self.loss)
            train_image = tf.summary.image(name='train_input',
                                           tensor= tf.reshape(self.input, shape=(-1, 28, 28, 1)),
                                           max_outputs=self.params.max_output)
            # tf.summary.image(name='train_encoded', tensor= tf.reshape(self.rep, shape=(-1, 28, 28, 1)), max_outputs=5)
            train_decode = tf.summary.image(name='train_decoded',
                                            tensor= tf.reshape(self.decoded_op, shape=(-1, 28, 28, 1)),
                                            max_outputs=self.params.max_output)
            self.merged_summary_train = tf.summary.merge([train_loss, train_image, train_decode])
        elif self.params.mode == 'VA':
            valid_loss = tf.summary.scalar('valid_loss', self.loss)
            valid_image = tf.summary.image(name='valid_input',
                                           tensor=tf.reshape(self.input, shape=(-1, 28, 28, 1)),
                                           max_outputs=self.params.max_output)
            # tf.summary.image(name='valid_encoded', tensor=tf.reshape(self.rep, shape=(-1, 28, 28, 1)), max_outputs=5)
            valid_decode = tf.summary.image(name='valid_decoded',
                                            tensor=tf.reshape(self.decoded_op, shape=(-1, 28, 28, 1)),
                                            max_outputs=self.params.max_output)
            self.merged_summary_valid = tf.summary.merge([valid_loss, valid_image, valid_decode])

    def train(self):
        global optimizer
        with tf.variable_scope('optimize_tar_net'):
            learning_rate = self.params.learning_rate
            self.compute_loss()
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
