#!/bin/bash

mkdir data

# data download
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en -P ./data
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de -P ./data

# vocab download
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en -P ./data
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de -P ./data

# dictionary download
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/dict.en-de -P ./data

# test download
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en -P ./data
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de -P ./data
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en -P ./data
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de -P ./data
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en -P ./data
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de -P ./data
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en -P ./data
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de -P ./data
