import os


class Directory():
    def __init__(self, mode):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """

        self.mode = mode

        self.root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.utility_dir = self.root_path + '/utility_dir'
        self.curr_utility_dir = self.utility_dir + '/mnist'
        self.preprocessing_dir = self.root_path + '/pre_processing'

        '''Directory to utility paths'''
        self.data_path = self.curr_utility_dir + '/data'
        self.vocab_path = self.curr_utility_dir + '/vocab'
        self.model_path = self.curr_utility_dir + '/models'
        self.output_path = self.curr_utility_dir + '/output'
        self.log_path = self.curr_utility_dir + '/log_dir'
        self.log_emb_path = self.log_path + '/emb_viz'

        self.makedir(self.vocab_path)
        self.makedir(self.model_path)
        self.makedir(self.output_path)
        self.makedir(self.log_path)
        self.makedir(self.log_emb_path)

        self.glove_path = self.data_path + '/glove_dict.pkl'

        '''Directory to dataset'''

        self.data_filename = self.data_path + '/train-images-idx3-ubyte.gz'
        self.label_filename = self.data_path + '/train-labels-idx1-ubyte.gz'

        if (mode == 'VA'):
            # self.data_filename = self.data_path + '/tokenized_valid.txt'
            # self.label_filename = self.data_path + '/label_valid.txt'
            self.data_filename = None   # for MNIST, sampling is done from train set itself
            self.label_filename = None  # for MNIST, sampling is done from train set itself
        elif (mode == 'TE'):
            self.data_filename = self.data_path + '/t10k-images-idx3-ubyte.gz'
            self.label_filename = self.data_path + '/t10k-labels-idx3-ubyte.gz'

        ''' ****************** Directory to saving or loading a model ********************** '''''
        self.latest_checkpoint = 'checkpoint'
        self.model_name = '/cnn_autoencoder_d50.ckpt'  # model name .ckpt is the model extension
        ''' ********** ********* ******** ********* ********* ********* ******** ************* '''''

        '''Directory to csv and pkl files'''
        self.vocab_size_file = self.vocab_path + '/vocab_size.txt'
        self.word_embedding = self.vocab_path + '/word_embedding.csv'
        self.word_vocab_dict = self.vocab_path + '/word_vocab.pkl'
        self.glove_present_training_word_vocab = self.vocab_path + '/glove_present_training_word_vocab.pkl'
        self.label_map_dict = self.vocab_path + '/label_map.pkl'

        ''' ****************** Directory for test model ********************** '''''
        self.test_model_name = '/cnn_autoencoder_d50.ckpt'
        self.test_model = self.model_path + self.test_model_name

    def makedir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
