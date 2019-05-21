# -*- coding: utf-8 -*-

import io
import math
from collections import Counter
from tqdm import tqdm
import tensorflow as tf

FILE_PATH = './data/'

def create_dataset(path, limit_size=None):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    lines = [line for line in tqdm(lines[:limit_size])]

    # Print examples
    for line in lines[:5]:
        print(line)

    return lines 

def create_dataset_test(path, lang=['en', 'de']):
    dataset_train_input_path = 'train.{}'.format(lang[0]) 
    dataset_train_target_path = 'train.{}'.format(lang[1])
    in_sent = create_dataset(path + dataset_train_en_path, 50000)
    print(in_sent)

def tokenize(text, vocab, max_len):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')

    lang_tokenizer.word_index = vocab

    tensor = lang_tokenizer.texts_to_sequences(text)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_len, padding='post')

    return tensor, lang_tokenizer

def load_dataset(path, max_len, limit_size=None, lang=['en', 'de']):
    dataset_train_input_path = 'train.{}'.format(lang[0]) 
    dataset_train_target_path = 'train.{}'.format(lang[1])

    print('Loading...')
    vocab_input = load_vocab(path, lang[0])
    vocab_target = load_vocab(path, lang[1])
    
    input_text = create_dataset(path + dataset_train_input_path, limit_size)
    target_text = create_dataset(path + dataset_train_target_path,limit_size)

    input_text = ['<s> ' + text + ' </s>' for text in target_text]
    target_text = [text + ' </s>' for text in target_text]

    input_tensor, input_lang_tokenizer = tokenize(input_text, vocab_input, max_len)
    target_tensor, target_lang_tokenizer = tokenize(target_text, vocab_target, max_len)

    return input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer

def max_length(tensor):
    return max(len(t) for t in tensor)
    
def load_dataset_test(path):

    it, tt, ilt, tlt =  load_dataset(path, 90, 5000)
    print(tt[0].shape)
    print(it.shape, tt.shape)
    max_it, max_tt = max_length(it), max_length(tt)
    print(max_it, max_tt)

def load_vocab(path, lang):
    lines = io.open(path + 'vocab.50K.{}'.format(lang), encoding='UTF-8').read().strip().split('\n')
    vocab = {}
    
    # 0 is padding
    for idx, word in enumerate(lines):
        vocab[word] = idx + 1

    vocab['<eos>'] = len(vocab) + 1

    return vocab

def convert_vocab(tokenizer, vocab):
    for key, val in vocab.items():
        tokenizer.index_word[val] = key

def select_optimizer(optimizer, learning_rate):
    if optimizer == 'adam':
        return tf.optimizers.Adam(learning_rate)
    elif optimizer == 'sgd':
        return tf.optimizers.SGD(learning_rate)
    elif optimizer == 'rmsprop':
        return tf.optimizers.RMSprop(learning_rate)

def loss_function(loss_object, y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = loss_object(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)

def ngrams(text, n):
    """
    Argus:
        text - list type, Ex. ['I', 'like', 'a', 'dog', '.']
        n - Divide by n, int type 
    """
    if type(text)==str: text=text.split()
    grams = (tuple(text[idx:idx+n]) for idx in range(len(text)-n+1))
    return grams

class BLEU():
    """ref: http://www.nltk.org/_modules/nltk/align/bleu.html
    """
    @staticmethod
    def compute(candidate, references, weights):
        """
        Argus:
            candidate - list type
            references - dual list type
            weights - list type
        """
        candidate = [word.lower() for word in candidate]
        references = [[word.lower() for word in reference] for reference in references]

        p_ns = (BLEU.modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1))
        print(p_ns)
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
        
        bp = BLEU.brevity_penalty(candidate, references)

        return bp * math.exp(s)

    @staticmethod
    def modified_precision(candidate, references, n):
        counts = Counter(ngrams(candidate, n))

        if len(counts)==0: return 0

        max_counts = {}
        for reference in references:
            reference_counts = Counter(ngrams(reference, n))
            for ngram in counts:
                max_counts[ngram] = max(max_counts.get(ngram,0), reference_counts[ngram])

        clipped_counts = dict((ngram, min(cnt, max_counts[ngram])) for ngram, cnt in counts.items())

        return sum(clipped_counts.values()) / sum(counts.values()) 

    @staticmethod
    def brevity_penalty(candidate, references):
        c = len(candidate)
        r = min(abs(len(r) - c) for r in references)

        if c > r:
            return 1
        else:
            return math.exp(1 - r / c)
        

def main():
    #load_dataset_test(FILE_PATH)
    weights = [0.25, 0.25, 0.25, 0.25]
    candidate1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
                    'ensures', 'that', 'the', 'military', 'always',
                    'obeys', 'the', 'commands', 'of', 'the', 'party']
    reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
                   'ensures', 'that', 'the', 'military', 'will', 'forever',
                   'heed', 'Party', 'commands']

    reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
                   'guarantees', 'the', 'military', 'forces', 'always',
                   'being', 'under', 'the', 'command', 'of', 'the',
                   'Party']

    reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
                   'army', 'always', 'to', 'heed', 'the', 'directions',
                   'of', 'the', 'party']
    print(ngrams(candidate1, 3))
    print(BLEU.compute(candidate1, [reference1, reference2, reference3], weights))

   
    pass

if __name__=='__main__':
    main()
