# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace
from run import train, test

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
    parser.add_argument('--learning-rate', help='learning rate <default: 0.001>', 
                        metavar='REAL', type=float, default=0.001)
    parser.add_argument('--dropout', help='dropout probability <default: 0>', 
                        metavar='REAL', type=float, default=.0)
    parser.add_argument('--method', help='content-based function <default: concat>', 
                        metavar='STRING', default='concat')

    parser.add_argument('--gpu-num', help='GPU number to use <default: 0>', 
                        metavar='INT', type=int, default=0)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)

if __name__=='__main__':
    main()
