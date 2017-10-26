#! /usr/bin/env python3
import os
import argparse
import datetime
import torch
import torchtext.data as data
import model
import train
import sys
import json
from util import sst, mr
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-predictfile', type=str, default=None, help='predict sentences in a file')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument('-dataset', type=str, default='sst', help='specify dataset: sst | mr')
    parser.add_argument('-fine-grained', action='store_true', default=False, help='use 5-class sst')
    args = parser.parse_args()

    # load data
    print("\nLoading data...")

    split_args = {
          'batch_size': args.batch_size,
          'fine_grained': args.fine_grained,
          'device': args.device,
          'repeat': False,
          'shuffle':  args.shuffle if args.shuffle else None
          }

    if args.dataset == 'mr':
       text_field, label_field, train_iter, dev_iter = mr(**split_args)
    elif args.dataset == 'sst':
       text_field, label_field, train_iter, dev_iter, test_iter = sst(**split_args)

    # update args and print
    args.embed_num = len(text_field.vocab)
    args.class_num = len(label_field.vocab) - 1 # exclude <unk>
    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # model
    if args.snapshot is None:
        cnn = model.CNN_Text(args)
    else:
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            cnn = torch.load(args.snapshot)
        except:
            print("Sorry, This snapshot doesn't exist."); exit()

    if args.cuda:
        cnn = cnn.cuda()

    # train or predict
    if args.predict is not None:
        label = train.predict(args.predict, cnn, text_field, label_field, args)
        print('\n[Text] {} [Label] {}\n'.format(args.predict, label))
    elif args.predictfile is not None:
        filepre = os.path.splitext(os.path.basename(args.predictfile))[0]
        result_path = os.path.join(os.path.dirname(os.path.realpath(args.predictfile)), filepre + '-predictions.json')
        with open(args.predictfile, 'r') as rf, \
             open(result_path, 'w') as wf:
           results = []
           print()
           for i, line in enumerate(rf):
              line = line.strip()
              label = train.predict(line, cnn, text_field, label_field, args)
              results.append(OrderedDict([('text', line), ('label', label)]))
              sys.stdout.write('\rPredicted [{}] sentences...'.format(i + 1))
           print('\nPredictions are written to ', result_path)
           json.dump(results, wf, indent=4)

    elif args.test:
        try:
            train.eval(test_iter, cnn, args)
        except Exception as e:
            print("\nSorry. The test dataset doesn't exist.\n")
    else:
        print()
        train.train(train_iter, dev_iter, cnn, args)

main()
