# -*- coding: utf-8 -*-

import os
import json
import time

from argparse import ArgumentParser, Namespace
from sklearn.model_selection import train_test_split

import tensorflow as tf

from data_loader import load_dataset  
from model import Encoder, Decoder, AttentionLayer

def test(args: Namespace):
    cfg = json.load(open(args.config_path, 'r', encoding='UTF-8'))
#    for key, val in cfg.items():
#        setattr(cfg, key, val)
    print(cfg['mode'])
    pass


def train(args: Namespace):
    input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer = load_dataset('./data/', args.max_len, 5000)

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

    setattr(args, 'max_len_input', max_len_input)
    setattr(args, 'max_len_target', max_len_target)

    setattr(args, 'steps_per_epoch', steps_per_epoch)
    setattr(args, 'vocab_input_size', vocab_input_size)
    setattr(args, 'vocab_target_size', vocab_target_size)
    setattr(args, 'BUFFER_SIZE', BUFFER_SIZE)

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)

    print('dataset shape (batch_size, max_len):', dataset)
    
    encoder = Encoder(vocab_input_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)

    if args.optimizer == 'adam':
        optimizer = tf.optimizers.Adam(learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = tf.optimizers.SGD(learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer = tf.optimizers.RMSprop(learning_rate)

    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        _loss = loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=_loss.dtype)
        _loss *= mask

        return tf.reduce_mean(_loss)

    @tf.function
    def train_step(_input, _target, enc_state):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_state = encoder(_input, enc_state)

            dec_hidden = enc_state

            dec_input = tf.expand_dims([target_lang_tokenizer.word_index['<eos>']] * BATCH_SIZE, 1)

            # First input feeding definition
            h_t = tf.zeros((BATCH_SIZE, 1, embedding_dim))

            for idx in range(1, _target.shape[1]):
                # idx means target character index.
                predictions, dec_hidden, h_t = decoder(dec_input, dec_hidden, enc_output, h_t)

                loss += loss_function(_target[:, idx], predictions)

                dec_input = tf.expand_dims(_target[:, idx], 1)

        batch_loss = (loss / int(_target.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


    # Setting checkpoint
    now = time.localtime(time.time())
    now_time = '/{}{}{}{}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    checkpoint_dir = './training_checkpoints' + now_time
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    
    min_total_loss = 1000

    for epoch in range(EPOCHS):
        start = time.time()

#        if epoch+1 >= 5:
#            learning_rate /= 2

        enc_hidden = encoder.initialize_hidden_state()
        enc_cell = encoder.initialize_cell_state()
        enc_state = [[enc_hidden, enc_cell], [enc_hidden, enc_cell], [enc_hidden, enc_cell], [enc_hidden, enc_cell]]

        total_loss = 0

        for(batch, (_input, _target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(_input, _target, enc_state)
            total_loss += batch_loss

            if batch % 10 == 0:
                print('Epoch {}/{} Batch {}/{} Loss {:.4f}'.format(epoch + 1,
                                                             EPOCHS,
                                                             batch + 10,
                                                             steps_per_epoch,
                                                             batch_loss.numpy()))


        print('Epoch {}/{} Total Loss per epoch {:.4f} - {} sec'.format(epoch + 1, 
                                                             EPOCHS,
                                                             total_loss / steps_per_epoch,
                                                             time.time() - start))

        # saving checkpoint 
        if min_total_loss > total_loss / steps_per_epoch:
            print('Saving checkpoint...')
            min_total_loss = total_loss / steps_per_epoch
            checkpoint.save(file_prefix= checkpoint_prefix)

        print('\n')

    # saving a information of the model
    with open('{}/config.json'.format(checkpoint_dir), 'w', encoding='UTF-8') as fout:
        json.dump(vars(args), fout, indent=2, sort_keys=True)


def main():

    parser = ArgumentParser(description='train model from data')

    parser.add_argument('--mode', help='train or test', metavar='MODE',
                        default='train')

    parser.add_argument('--config-path', help='config json path', metavar='DIR')
    
    parser.add_argument('--init-checkpoint', help='checkpoint file', 
                        metavar='FILE')

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
                        metavar='INT', type=int, default=0.001)

    parser.add_argument('--gpu-num', help='GPU number to use <default: 0>', 
                        metavar='INT', type=int, default=0)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


if __name__=='__main__':
    main()
