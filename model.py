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

    def call(self, x, pre_state):
        x = self.embedding(x)
        x, state_h_1, state_c_1 = self.lstm_1(x, initial_state=pre_state[0])
        x, state_h_2, state_c_2 = self.lstm_2(x, initial_state=pre_state[1])
        x, state_h_3, state_c_3 = self.lstm_3(x, initial_state=pre_state[2])
        output, state_h_4, state_c_4 = self.lstm_4(x, initial_state=pre_state[3])
        state = [[state_h_1, state_c_1], [state_h_2, state_c_2], [state_h_3, state_c_3], [state_h_4, state_c_4]]
        return output, state

    def initialize_hidden_state(self):
        return tf.random.uniform((self.batch_size, self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.lstm_1 = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        
        self.lstm_2 = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

        self.lstm_3 = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

        self.lstm_4 = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

        self.attention_layer = AttentionLayer(units)

        self.W_c = tf.keras.layers.Dense(embedding_dim, activation='tanh')

        self.W_s = tf.keras.layers.Dense(vocab_size)



    def call(self, x, pre_state, enc_output, pre_h_t):
        x = self.embedding(x)

        # input_feeding shape == (batch_size, 1, word_embedding_dim + pre_h_t_embedding_dim)
        x = tf.concat([x, pre_h_t], axis=-1)
        x, state_h_1, state_c_1 = self.lstm_1(x, initial_state=pre_state[0])
        x, state_h_2, state_c_2 = self.lstm_2(x, initial_state=pre_state[1])
        x, state_h_3, state_c_3 = self.lstm_3(x, initial_state=pre_state[2])

        # dec_output shape == (batch_size, 1, units)
        dec_output, state_h_4, state_c_4 = self.lstm_4(x, initial_state=pre_state[3])
        
        state = [[state_h_1, state_c_1], [state_h_2, state_c_2], [state_h_3, state_c_3], [state_h_4, state_c_4]]

        context_vector = self.attention_layer(dec_output, enc_output)

        # h_t shape == (batch_size, embedding_dim)
        h_t = self.W_c(tf.concat([tf.expand_dims(context_vector, 1), dec_output], axis=-1))
        #h_t = self.W_c(tf.concat([context_vector, tf.squeeze(dec_output)], axis=-1))

        # y_t shape == (batch_size, vocab_size)
        y_t = tf.squeeze(self.W_s(h_t))

        return y_t, state, h_t 


class AttentionLayer(tf.keras.Model):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W_a = tf.keras.layers.Dense(units)
        self.v_a = tf.keras.layers.Dense(1)

    def call(self, dec_h_t, enc_h_s):

#        print('dec_h_t shape:', dec_h_t.shape)
#        print('enc_h_s shape:', enc_h_s.shape)
        # dec_h_t shape == (batch_size, 1, units)
        # enc_h_s shape == (batch_size, seq_len, units) 
#        concat_h = tf.concat([dec_h_t, enc_h_s], axis=1)
#        concat_h = tf.reshape(concat_h, [concat_h.shape[0] * concat_h.shape[1], concat_h.shape[2]])
#        print('concat_h shape:', concat_h.shape)
        score = self.v_a(tf.nn.tanh(self.W_a(dec_h_t + enc_h_s)))
#        print('score shape:', score.shape)

        # attention_weights shape == (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
#        print('attention_weights shape:', attention_weights.shape)

        # context_vector shape == (batch_size, units)
        context_vector = tf.reduce_sum(tf.matmul(attention_weights, enc_h_s, transpose_a=True), axis=1)

#        print('context_vector shape:', context_vector.shape)

        return context_vector

        


def main():
    pass

if __name__=='__main__':
    main()
