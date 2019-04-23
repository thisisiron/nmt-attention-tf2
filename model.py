# -*- coding: utf-8 -*-

import tensorflow as tf


class Embedder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    def call(self, input_tensor):
        return self.embedding(input_tensor)


class Encoder(tf.keras.Model):
    def __init__(self, cells, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_cells = cells
        self.lstm = tf.keras.layers.LSTM(self.enc_cells,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        output, state = self.lstm(x)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_cells))


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def call(self):
        pass

class AttentionLayer(tf.keras.Model):
    def __init__(self):
        super(Attention, self).__init__()
        pass

    def call(self):
        pass


def main():
    pass

if __name__=='__main__':
    main()
