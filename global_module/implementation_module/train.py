import numpy as np
from global_module.implementation_module import autoencoder
from global_module.implementation_module import reader

class Train:

    def run_epoch(self, session, model_obj, params, input):

        for step, curr_input in enumerate(reader(params).data_iterator(input)):
            feed_dict = {model_obj.input:curr_input}
            loss = session.run([model_obj.loss],
                               feed_dict=feed_dict)