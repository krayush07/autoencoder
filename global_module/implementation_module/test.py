import numpy as np
from global_module.implementation_module import Autoencoder
from global_module.implementation_module import Reader
import tensorflow as tf
from global_module.settings_module import ParamsClass, Directory, Dictionary
import random
import sys
import time


class Test:
    def __init__(self):
        self.iter_test = 0

    def run_epoch(self, session, min_loss, model_obj, reader, input, writer):
        global epoch_combined_loss, step
        params = model_obj.params
        epoch_combined_loss = 0.0

        output_file = open(model_obj.dir_obj.log_emb_path + '/latent_representation.csv', 'w')

        for step, curr_input in enumerate(reader.data_iterator(input)):
            feed_dict = {model_obj.input: curr_input}
            total_loss, latent_rep, summary_test = session.run([model_obj.loss, model_obj.rep, model_obj.merged_summary_test], feed_dict=feed_dict)

            epoch_combined_loss += total_loss

            self.iter_test += 1
            if self.iter_test % params.log_step == 0 and params.log:
                writer.add_summary(summary_test, self.iter_test)

            for each_rep in latent_rep:
                output_file.write(' '.join(str(x) for x in each_rep).strip() + '\n')

        epoch_combined_loss /= step
        output_file.close()
        return epoch_combined_loss, min_loss

    def run_test(self):
        global test_writer
        mode_test = 'TE'

        # test object
        params_test = ParamsClass(mode=mode_test)
        dir_test = Directory(mode_test)
        test_reader = Reader(params_test)
        test_instances = test_reader.read_image_data(dir_test.data_filename)

        random.seed(4321)

        global_min_loss = sys.float_info.max

        print('***** INITIALIZING TF GRAPH *****')

        with tf.Graph().as_default(), tf.Session() as session:
            with tf.variable_scope("model"):
                test_obj = Autoencoder(params_test, dir_test)

            model_saver = tf.train.Saver()
            model_saver.restore(session, test_obj.dir_obj.test_model)

            if params_test.log:
                test_writer = tf.summary.FileWriter(dir_test.log_path + '/test')

            print('**** TF GRAPH INITIALIZED ****')

            start_time = time.time()

            test_loss, _, = self.run_epoch(session, global_min_loss, test_obj, test_reader, test_instances, test_writer)
            print("Epoch: %d Test loss: %.4f" % (1, test_loss))

            curr_time = time.time()
            print('1 epoch run takes ' + str((curr_time - start_time) / 60) + ' minutes.')

            if params_test.log:
                test_writer.close()
