# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import tensorflow as tf
from data_loader import load_dataset  
from model import Embedder, Encoder, Decoder, AttentionLayer


FILE_PATH = './data/'
VAL_SPLIT = 0.1

def train():
    input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer = load_dataset(FILE_PATH)

    max_len_input = len(input_tensor[0])
    max_len_target = len(target_tensor[0])

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=VAL_SPLIT)

    # init
    BATCH_SIZE = 64
    steps_for_epoch = len(input_tensor_train) // BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_input_size = len(input_lang_tokenizer.word_index)+1
    vocab_target_size = len(target_lang_tokenizer.word_index)+1
    BUFFER_SIZE = len(input_tensor_train)

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)



def main():
    train()
    pass

if __name__=='__main__':
    main()
