# -*- coding: utf-8 -*-

import tensorflow as tf


class Embedder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    def call(self, input_tensor):
        return self.embedding(input_tensor)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm_1 = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

        self.lstm_2 = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

        self.lstm_3 = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

        self.lstm_4 = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

    def call(self, x, init_hidden, init_cell):
        x = self.embedding(x)
        x, _ = self.lstm_1(x, initial_state=[init_hidden, init_cell])
        x, _ = self.lstm_2(x)
        x, _ = self.lstm_3(x)
        output, state = self.lstm_4(x)
        return output, state

    def initialize_hidden_state(self, x):
        return tf.((self.batch_size, self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, cells, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_cells = cells
        self.lstm = tf.keras.layers.LSTM(self.dec_cells,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

    def call(self, x, init_hidden, enc_output):
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
