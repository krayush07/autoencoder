import gzip

import numpy as np

class DataReader:
    def __init__(self, params):
        self.params = params
        self.num_bytes = 4

    def read_bytes(self, bytestream, num_bytes):
        # big endian, 32 bit integer
        return np.frombuffer(bytestream.read(num_bytes), dtype=np.dtype(np.uint32).newbyteorder('B'))

    def read_image_data(self, filename):
        with gzip.open(filename) as bytestream:
            magic_number = self.read_bytes(bytestream, self.num_bytes)
            num_images = self.read_bytes(bytestream, self.num_bytes)[0]
            num_rows = self.read_bytes(bytestream, self.num_bytes)[0]
            num_cols = self.read_bytes(bytestream, self.num_bytes)[0]
            data_buffer = bytestream.read(num_images * num_rows * num_cols)
            data = np.frombuffer(data_buffer, dtype=np.uint8).astype(np.float32).reshape(num_images, num_rows, num_cols, 1)
            return data

    def read_image_label(self, filename):
        with gzip.open(filename) as bytestream:
            magic_number = self.read_bytes(bytestream, self.num_bytes)
            num_images = self.read_bytes(bytestream, self.num_bytes)[0]
            data_buffer = bytestream.read(num_images)
            data = np.frombuffer(data_buffer, dtype=np.uint8).astype(np.float32).reshape(num_images, 1)
            return data

    def data_iterator(self, input):
        num_batches = len(input) // self.params.batch_size

        input = input.reshape(len(input), -1)
        for i in range(50):
            yield input[i * self.params.batch_size : (i+1) * self.params.batch_size]

# def main():
#     test_labels = DataReader(None).read_image_label('/autoencoder/global_module/utility_dir/mnist/data/t10k-labels-idx1-ubyte.gz')
#     for each_label in test_labels:
#         print int(each_label[0])
#     print test_labels
#
# if __name__ == '__main__':
#     main()