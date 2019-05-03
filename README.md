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
               [--optimizer STRING] [--learning_rate INT] [--method STRING]
               [--gpu-num INT]

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
  --learning_rate INT   learning_rate <default: 1>
  --method STRING       content-based function <default: concat>
```

Train command example
```
python main.py --max-len 50 --embedding-dim 100 --batch-size 60 --method concat
```

Test command example
```
python main.py --mode test --config-path training_checkpoints/{TRAINING_CHECKPOINT}
```

## Results
|         | Train Set ACC    | Validation Set ACC    | Test Set ACC |
|---------|------------------|-----------------------|--------------|
| Model   | --%              | --%                   | --%          |

## Reference
[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025?context=cs)<br>
