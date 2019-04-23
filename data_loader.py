# -*- coding: utf-8 -*-

import io
from tqdm import tqdm
import tensorflow as tf

FILE_PATH = './data/'

def create_dataset(path, limit_size=None):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    lines = [line + ' <eos>' for line in tqdm(lines[:limit_size])]

    print(lines[:5])

    return lines 

def create_dataset_test(path, lang=['en', 'de']):
    dataset_train_input_path = 'train.{}'.format(lang[0]) 
    dataset_train_target_path = 'train.{}'.format(lang[1])
    in_sent = create_dataset(path + dataset_train_en_path, 50000)
    print(in_sent)

def tokenize(text, vocab):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    lang_tokenizer.word_index = vocab

    tensor = lang_tokenizer.texts_to_sequences(text)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer

def load_dataset(path, lang=['en', 'de']):
    dataset_train_input_path = 'train.{}'.format(lang[0]) 
    dataset_train_target_path = 'train.{}'.format(lang[1])

    print('Loading...')
    vocab_input = load_vocab(path, lang[0])
    vocab_target = load_vocab(path, lang[1])
    
    input_text = create_dataset(path + dataset_train_input_path)
    target_text = create_dataset(path + dataset_train_target_path)

    input_tensor, input_lang_tokenizer = tokenize(input_text, vocab_input)
    target_tensor, target_lang_tokenizer = tokenize(target_text, vocab_target)

    return input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer
    
def load_dataset_test(path):
    it, tt, ilt, tlt =  load_dataset(path)
    print(it[0])
    print(tt[0])
    print('input_len', len(it[0]))
    print('target_len', len(tt[0]))

def load_vocab(path, lang):
    lines = io.open(path + 'vocab.50K.{}'.format(lang), encoding='UTF-8').read().strip().split('\n')
    vocab = {}

    for idx, word in enumerate(lines):
        vocab[word] = idx

    vocab['<eos>'] = len(vocab)

    return vocab

def main():
    pass

if __name__=='__main__':
    main()
