# nmt-attention-tf
Effective Approaches to Attention-based Neural Machine Translation implemented as Tensorflow 2.0

## Requirements
Tensorflow == 2.0_alpha <br>
Python == 3.6

## Data
WMT'14 English-German data: https://nlp.stanford.edu/projects/nmt/

Download the datasets using the following script:
```
./download.sh
```

## Usage

```
usage: main.py [-h] [--mode MODE] [--config-path DIR] [--init-checkpoint FILE]
               [--batch-size INT] [--epoch INT] [--embedding-dim INT]
               [--max-len INT] [--units INT] [--dev-split REAL]
               [--optimizer STRING] [--learning-rate REAL] [--dropout REAL]
               [--method STRING] 

train model from data

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           train or test
  --config-path DIR     config json path
  --init-checkpoint FILE
                        checkpoint file
  --batch-size INT      batch size <default: 32>
  --epoch INT           epoch number <default: 10>
  --embedding-dim INT   embedding dimension <default: 256>
  --max-len INT         max length of a sentence <default: 90>
  --units INT           units <default: 512>
  --dev-split REAL      <default: 0.1>
  --optimizer STRING    optimizer <default: adam>
  --learning-rate REAL  learning rate <default: 0.001>
  --dropout REAL        dropout probability <default: 0>
  --method STRING       content-based function <default: concat>
```

Train command example
```
python main.py --max-len 50 --embedding-dim 100 --batch-size 60 --method concat
```

Test command example
```
python main.py --mode test --config-path training_checkpoints/{TRAINING_CHECKPOINT}/config.json
```

## Demo
I think this demo is poor performance because I don't have a large resource. So, The paper proposed embedding dimension sets 1000. But this demo's embedding dimension is 50. And this is trained only for 4 epochs.

If you don't have training_checkpoints directory, make training_checkpoints directory and proceed with the next step.

```
mkdir training_checkpoints
cd training_checkpoints
```

You can download [here](https://drive.google.com/open?id=19VtPQ-9gyLkNxRjD7GbjACaTM5xAz_lH). And you put DEMO directory in training_checkpoints directory.

```
python main.py --mode test --config-path training_checkpoints/DEMO/config.json
```

Example
```
Input Sentence or If you want to quit, type Enter Key : Where are you?
Early stopping
<s> wo sind sie ? </s> 
<s> where are you ? </s>
```

## Results
|         | Train Set BLEU    | Test Set BLEU |
|---------|-------------------|---------------|
| Model   | --                | --            |

## Reference
[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025?context=cs)<br>
