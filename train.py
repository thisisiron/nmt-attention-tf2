# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time

from argparse import ArgumentParser, Namespace
from sklearn.model_selection import train_test_split

import tensorflow as tf

from utils import load_dataset, load_vocab, convert_vocab, select_optimizer, loss_function
from model import Encoder, Decoder, AttentionLayer

def test(args: Namespace):
    cfg = json.load(open(args.config_path, 'r', encoding='UTF-8'))

    encoder = Encoder(cfg['vocab_input_size'], cfg['embedding_dim'], cfg['units'], 1)
    decoder = Decoder(cfg['vocab_target_size'], cfg['embedding_dim'], cfg['units'], cfg['method'], 1)
    optimizer = select_optimizer(cfg['optimizer'], cfg['learning_rate']) 

    ckpt = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    manager = tf.train.CheckpointManager(ckpt, cfg['checkpoint_dir'], max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    while True:
        sentence = input('Input Sentence or If you want to quit, type Enter Key : ')

        if sentence == '': break

        sentence = re.sub(r"(\.\.\.|[?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)

        sentence = '<s> ' + sentence.lower().strip() + ' </s>'

        input_vocab = load_vocab('./data/', 'en')
        target_vocab = load_vocab('./data/', 'de')

        input_lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
        input_lang_tokenizer.word_index = input_vocab 

        target_lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
        target_lang_tokenizer.word_index = target_vocab

        convert_vocab(input_lang_tokenizer, input_vocab)
        convert_vocab(target_lang_tokenizer, target_vocab)


        inputs = [input_lang_tokenizer.word_index[i] if i in input_lang_tokenizer.word_index else input_lang_tokenizer.word_index['<unk>'] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=cfg['max_len_input'],
                                                               padding='post')

        inputs = tf.convert_to_tensor(inputs)

        result = ''

        enc_hidden = encoder.initialize_hidden_state()
        enc_cell = encoder.initialize_cell_state()
        enc_state = [[enc_hidden, enc_cell], [enc_hidden, enc_cell], [enc_hidden, enc_cell], [enc_hidden, enc_cell]]

        enc_output, enc_hidden = encoder(inputs, enc_state)

        dec_hidden = enc_hidden
        #dec_input = tf.expand_dims([target_lang_tokenizer.word_index['<eos>']], 0)
        dec_input = tf.expand_dims([target_lang_tokenizer.word_index['<s>']], 0)

        h_t = tf.zeros((1, 1, cfg['embedding_dim']))

        for t in range(int(cfg['max_len_target'])):
            predictions, dec_hidden, h_t = decoder(dec_input,
                                                   dec_hidden,
                                                   enc_output,
                                                   h_t)

            # predeictions shape == (1, 50002)

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += target_lang_tokenizer.index_word[predicted_id] + ' '

            if target_lang_tokenizer.index_word[predicted_id] == '</s>':
                print('Early stopping')
                break

            dec_input = tf.expand_dims([predicted_id], 0)

        print(result)
        print(sentence)
        sys.stdout.flush()


def predict(args: Namespace):
    cfg = json.load(open(args.config_path, 'r', encoding='UTF-8'))


def train(args: Namespace):
    input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer = load_dataset('./data/', args.max_len)

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
    decoder = Decoder(vocab_target_size, embedding_dim, units, args.method, BATCH_SIZE)

    optimizer = select_optimizer(args.optimizer, args.learning_rate)

    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


    @tf.function
    def train_step(_input, _target, enc_state):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_state = encoder(_input, enc_state)

            dec_hidden = enc_state

            dec_input = tf.expand_dims([target_lang_tokenizer.word_index['<s>']] * BATCH_SIZE, 1)

            # First input feeding definition
            h_t = tf.zeros((BATCH_SIZE, 1, embedding_dim))

            for idx in range(1, _target.shape[1]):
                # idx means target character index.
                predictions, dec_hidden, h_t = decoder(dec_input, 
                                                       dec_hidden, 
                                                       enc_output, 
                                                       h_t)


                loss += loss_function(loss_object, _target[:, idx], predictions)

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
    setattr(args, 'checkpoint_dir', checkpoint_dir) 
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    # saving information of the model
    with open('{}/config.json'.format(checkpoint_dir), 'w', encoding='UTF-8') as fout:
        json.dump(vars(args), fout, indent=2, sort_keys=True)
    
    min_total_loss = 1000

    for epoch in range(EPOCHS):
        start = time.time()

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



def main():
    pass



if __name__=='__main__':
    main()
