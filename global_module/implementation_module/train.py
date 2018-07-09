import numpy as np
from global_module.implementation_module import Autoencoder
from global_module.implementation_module import Reader
import tensorflow as tf
from global_module.settings_module import ParamsClass, Directory, Dictionary
import random
import sys
import time


class Train:
    def __init__(self):
        self.iter_train = 0
        self.iter_valid = 0

    def run_epoch(self, session, min_loss, model_obj, reader, input, writer):
        global epoch_combined_loss, step
        epoch_combined_loss = 0.0

        params = model_obj.params
        dir_obj = model_obj.dir_obj
        for step, curr_input in enumerate(reader.data_iterator(input)):
            feed_dict = {model_obj.input: curr_input}
            if params.mode == 'TR':
                run_metadata = tf.RunMetadata()
                inp, op, total_loss, summary_train, _ = session.run([model_obj.input, model_obj.decoded_op, model_obj.loss, model_obj.merged_summary_train, model_obj.train_op],
                                                                    run_metadata=run_metadata,
                                                                    feed_dict=feed_dict)
                self.iter_train += 1
                if self.iter_train % params.log_step == 0 and params.log:
                    writer.add_run_metadata(run_metadata, 'step%d' % self.iter_train)
                    writer.add_summary(summary_train, self.iter_train)

            elif params.mode == 'VA':
                total_loss, summary_valid = session.run([model_obj.loss, model_obj.merged_summary_valid],
                                                        feed_dict=feed_dict)
                self.iter_valid += 1
                if self.iter_valid % params.log_step == 0 and params.log:
                    writer.add_summary(summary_valid, self.iter_valid)

            else:
                total_loss = session.run(model_obj.loss, feed_dict=feed_dict)

            epoch_combined_loss += total_loss

        epoch_combined_loss /= step
        if params.mode == 'VA':
            model_saver = tf.train.Saver()
            print('**** Current minimum on valid set: %.4f ****' % min_loss)

            if epoch_combined_loss < min_loss:
                min_loss = epoch_combined_loss
                model_saver.save(session,
                                 save_path=dir_obj.model_path + dir_obj.model_name,
                                 latest_filename=dir_obj.latest_checkpoint)
                print('==== Model saved! ====')

        return epoch_combined_loss, min_loss

    def run_train(self):
        global train_writer, valid_writer
        mode_train, mode_valid = 'TR', 'VA'

        # train object
        params_train = ParamsClass(mode=mode_train)
        dir_train = Directory(mode_train)
        train_reader = Reader(params_train)
        train_instances = train_reader.read_image_data(dir_train.data_filename)

        # valid object
        params_valid = ParamsClass(mode=mode_valid)
        dir_valid = Directory(mode_valid)
        valid_reader = Reader(params_valid)
        if dir_valid.data_filename is None:
            all_instances = train_instances
            train_instances = all_instances[: int(0.8 * len(all_instances))]
            valid_instances = all_instances[int(0.8 * len(all_instances)):]
        else:
            valid_instances = valid_reader.read_image_data(dir_valid.data_filename)

        random.seed(4321)
        if (params_train.enable_shuffle):
            random.shuffle(train_instances)
            random.shuffle(valid_instances)

        global_min_loss = sys.float_info.max

        print('***** INITIALIZING TF GRAPH *****')

        with tf.Graph().as_default(), tf.Session() as session:
            # random_normal_initializer = tf.random_normal_initializer()
            # random_uniform_initializer = tf.random_uniform_initializer(-params_train.init_scale, params_train.init_scale)
            xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

            # with tf.name_scope('train'):
            with tf.variable_scope("model", reuse=None, initializer=xavier_initializer):
                train_obj = Autoencoder(params_train, dir_train)

            # with tf.name_scope('valid'):
            with tf.variable_scope("model", reuse=True, initializer=xavier_initializer):
                valid_obj = Autoencoder(params_valid, dir_valid)

            if not params_train.enable_checkpoint:
                session.run(tf.global_variables_initializer())

            if params_train.enable_checkpoint:
                ckpt = tf.train.get_checkpoint_state(dir_train.model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Loading model from: %s" % ckpt.model_checkpoint_path)
                    tf.train.Saver().restore(session, ckpt.model_checkpoint_path)

            print('**** TF GRAPH INITIALIZED ****')

            if params_train.log:
                train_writer = tf.summary.FileWriter(dir_train.log_path + '/train', session.graph)
                valid_writer = tf.summary.FileWriter(dir_train.log_path + '/valid')

            # train_writer.add_graph(tf.get_default_graph())

            start_time = time.time()
            for i in range(params_train.max_max_epoch):
                lr_decay = params_train.lr_decay ** max(i - params_train.max_epoch, 0.0)
                # train_obj.assign_lr(session, params_train.learning_rate * lr_decay)

                # print(params_train.learning_rate * lr_decay)

                print('\n++++++++=========+++++++\n')
                lr = params_train.learning_rate * lr_decay
                print("Epoch: %d Learning rate: %.5f" % (i + 1, lr))
                train_loss, _, = self.run_epoch(session, global_min_loss, train_obj, train_reader, train_instances, train_writer)
                print("Epoch: %d Train loss: %.4f" % (i + 1, train_loss))

                valid_loss, curr_min_loss = self.run_epoch(session, global_min_loss, valid_obj, valid_reader, valid_instances, valid_writer)
                if (curr_min_loss < global_min_loss):
                    global_min_loss = curr_min_loss

                print("Epoch: %d Valid loss: %.4f" % (i + 1, valid_loss))

                curr_time = time.time()
                print('1 epoch run takes ' + str(((curr_time - start_time) / (i + 1)) / 60) + ' minutes.')

            if params_train.log:
                train_writer.close()
                valid_writer.close()
