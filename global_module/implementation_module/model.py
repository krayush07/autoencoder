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
        self.normalized_input = tf.multiply(tf.constant(1.0/255.0), self.input)

    def autoencode(self):
        self.decoded_op, self.rep = self.model_utils.ffn_autoencoder(self.normalized_input, self.params.output_shape)

    def encode(self):
        self.rep = self.model_utils.ffn_encoder(self.normalized_input)

    def compute_loss(self):
        self.loss = tf.losses.mean_squared_error(self.normalized_input, self.decoded_op)

    def generate_summary(self):
        if self.params.mode == 'TR':
            loss_summary = 'train_loss'
            img_summary = 'train_input'
            dec_summary = 'train_decoded'
            train_decode, train_image, train_loss = self.collect_summary(loss_summary, img_summary, dec_summary)
            self.merged_summary_train = tf.summary.merge([train_loss, train_image, train_decode])

        elif self.params.mode == 'VA':
            loss_summary = 'valid_loss'
            img_summary = 'valid_input'
            dec_summary = 'valid_decoded'
            valid_decode, valid_image, valid_loss = self.collect_summary(loss_summary, img_summary, dec_summary)
            self.merged_summary_valid = tf.summary.merge([valid_loss, valid_image, valid_decode])

        elif self.params.mode == 'TE':
            loss_summary = 'test_loss'
            img_summary = 'test_input'
            dec_summary = 'test_decoded'
            test_decode, test_image, _ = self.collect_summary(loss_summary, img_summary, dec_summary)
            self.merged_summary_test = tf.summary.merge([test_image, test_decode])

    def collect_summary(self, loss_summary, img_summary, dec_summary):
        loss = tf.summary.scalar(loss_summary, self.loss)
        image = tf.summary.image(name=img_summary,
                                       tensor=tf.reshape(self.normalized_input, shape=(-1, 28, 28, 1)),
                                       max_outputs=self.params.max_output)
        decode = tf.summary.image(name=dec_summary,
                                        tensor=tf.reshape(self.decoded_op, shape=(-1, 28, 28, 1)),
                                        max_outputs=self.params.max_output)
        return decode, image, loss

    def train(self):
        global optimizer
        with tf.variable_scope('optimize_loss'):
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
