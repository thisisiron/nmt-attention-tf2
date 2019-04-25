# -*- coding: utf-8 -*-

import os

import time
from argparse import ArgumentParser, Namespace
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data_loader import load_dataset  
from model import Encoder, Decoder, AttentionLayer



def train(args: Namespace):
    input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer = load_dataset(args.data_path, args.max_len, 5000)

    max_len_input = len(input_tensor[0])
    max_len_target = len(target_tensor[0])

    print('max len of each seq:', max_len_input, ',', max_len_target)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=args.dev_split)

    # init hyperparameter
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size 
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    embedding_dim = args.embedding_dim 
    units = args.units 
    vocab_input_size = len(input_lang_tokenizer.word_index) + 1
    vocab_target_size = len(target_lang_tokenizer.word_index) + 1
    BUFFER_SIZE = len(input_tensor_train)
    learning_rate = args.learning_rate 

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)

    print('dataset shape: (batch_size, max_len):', dataset)
    
    encoder = Encoder(vocab_input_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)

    if args.optimizer == 'adam':
        optimizer = tf.optimizers.Adam()
    elif args.optimizer == 'sgd':
        optimizer = tf.optimizers.SGD()
    elif args.optimizer == 'rmsprop':
        optimizer = tf.optimizers.RMSprop()

    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(y_true, y_pred):
      mask = tf.math.logical_not(tf.math.equal(y_true, 0))
      _loss = loss_object(y_true, y_pred)

      mask = tf.cast(mask, dtype=_loss.dtype)
      _loss *= mask

      return tf.reduce_mean(_loss)





def main():

    parser = ArgumentParser(description='train model from data')

    parser.add_argument('--data-path', help='input data path prefix', 
                        metavar='NAME', default='./data/')
    parser.add_argument('--checkpoint-dir', help='checkpoint dir <default: ./training_checkpoints>', 
                        metavar='DIR', default=' ./training_checkpoints')

    parser.add_argument('--batch-size', help='batch size <default: 32>', metavar='INT', 
                        type=int, default=32)
    parser.add_argument('--epoch', help='epoch number <default: 10>', metavar='INT', 
                        type=int, default=10)
    parser.add_argument('--embedding-dim', help='embedding dimension <default: 256>', 
                        metavar='INT',type=int, default=256)
    parser.add_argument('--max-len', help='max length of a sentence <default: 90>', 
                        metavar='INT',type=int, default=90)
    parser.add_argument('--units', help='units <default: 512>', metavar='INT',
                        type=int, default=512)
    parser.add_argument('--dev-split', help='<default: 0.1>', metavar='REAL',
                        type=float, default=0.1)
    parser.add_argument('--optimizer', help='optimizer <default: adam>', 
                        metavar='STRING', default='adam')
    parser.add_argument('--learning_rate', help='learning_rate <default: 1>', 
                        metavar='INT', type=int, default=1)

    parser.add_argument('--gpu-num', help='GPU number to use <default: 0>', metavar='INT', type=int,
                        default=0)

    args = parser.parse_args()
    train(args)

if __name__=='__main__':
    main()
