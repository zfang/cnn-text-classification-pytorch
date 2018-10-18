#! /usr/bin/env python3
import argparse
import csv
import datetime
import os
import sys

import numpy as np
import torch

import model
import train
from util import sst, mr, load_word_vectors, headerless_tsv


def load_data(args):
    if args.dataset is None:
        return None, None, None, None, None

    # load data
    print("\nLoading data...")

    split_args = {
        'batch_size': args.batch_size,
        'device': args.device,
        'repeat': False,
        'shuffle': args.shuffle if args.shuffle else None
    }

    if args.dataset == 'mr':
        return mr(**split_args)
    elif args.dataset == 'sst':
        return sst(fine_grained=args.fine_grained, train_subtrees=args.train_subtrees, **split_args)
    else:
        return headerless_tsv(args.dataset, **split_args)


def main():
    parser = argparse.ArgumentParser(description='CNN text classifier')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-epochs', type=int, default=25, help='number of epochs for train')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training')
    parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status')
    parser.add_argument('-save-interval', type=int, default=0, help='how many steps to wait before saving')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-predictfile', type=str, default=None, help='predict sentences in a file')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument('-dataset', type=str, default='sst', help='specify dataset: sst | mr')
    parser.add_argument('-fine-grained', action='store_true', default=False, help='use 5-class sst')
    parser.add_argument('-train-subtrees', action='store_true', default=False, help='train sst subtrees')
    parser.add_argument('-debug', action='store_true', default=False, help='debug mode')
    args = parser.parse_args()

    # update args and print
    text_field, label_field, train_iter, dev_iter, test_iter = load_data(args)

    if train_iter:
        print("train dataset size:", len(train_iter.dataset))
    if dev_iter:
        print("dev dataset size:", len(dev_iter.dataset))
    if test_iter:
        print("test dataset size:", len(test_iter.dataset))

    if args.dataset:
        args.embed_num = len(text_field.vocab)
        args.class_num = len(label_field.vocab) - 1  # exclude <unk>

    word_vector_matrix = None
    if text_field:
        print("\nLoading pre-trained word vectors...")
        word_vector_matrix = load_word_vectors(vocab=text_field.vocab)

    args.cuda = (not args.no_cuda) and torch.cuda.is_available();
    del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    if args.dataset:
        args.save_dir = os.path.join(args.save_dir, args.dataset, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # model
    if args.snapshot is None:
        cnn = model.CNN_Text(args, text_field, label_field, word_vector_matrix)
    else:
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            cnn = torch.load(args.snapshot)
        except:
            print("Sorry, This snapshot doesn't exist.");
            exit()

    if args.cuda:
        cnn = cnn.cuda()

    print()

    # train or predict
    if args.predict is not None:
        label = train.predict(args.predict, cnn, args)
        print('[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    elif args.predictfile is not None:
        filepre = os.path.splitext(os.path.basename(args.predictfile))[0]
        predictions_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'predictions')
        result_path = os.path.join(predictions_dir, filepre + '-predictions-' + datetime.datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S') + '.csv')
        if not os.path.isdir(predictions_dir): os.makedirs(predictions_dir)
        with open(args.predictfile, 'r') as rf, \
                open(result_path, 'w') as wf:
            writer = csv.writer(wf)
            writer.writerow(['text', 'label'])
            for i, line in enumerate(rf):
                line = line.strip()
                label = train.predict(line, cnn, args)
                writer.writerow([line, label])
                sys.stdout.write('\rPredicted [{}] sentences...'.format(i + 1))
            print('\nPredictions are written to ', result_path)

    elif args.test:
        if test_iter:
            train.eval(test_iter, cnn, args)
        else:
            print("\nThe test dataset does not exist.\n")
    else:
        train.train(train_iter, dev_iter, cnn, args)
        print()


main()
