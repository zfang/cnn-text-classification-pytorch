import torchtext.data as data
import torchtext.datasets as datasets
from pymagnitude import *
from tqdm import tqdm

import mydatasets


def get_fields():
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    return text_field, label_field


def get_train_dev(callable, batch_size=32, **kargs):
    text_field, label_field = get_fields()

    train_data, dev_data = callable(text_field, label_field, **kargs)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, len(dev_data)),
        **kargs)

    return text_field, label_field, train_iter, dev_iter, None


def get_train_dev_test(callable, batch_size=32, **kargs):
    text_field, label_field = get_fields()

    train_data, dev_data, test_data = callable(text_field, label_field, **kargs)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(batch_size, len(dev_data), len(test_data)),
        **kargs)

    return text_field, label_field, train_iter, dev_iter, test_iter


# load MR dataset
def mr(shuffle=False, **kargs):
    return get_train_dev(
        lambda text_field, label_field, **ka: mydatasets.MR.splits(text_field, label_field, shuffle=shuffle),
        **kargs)


# load SST dataset
def sst(fine_grained=False, train_subtrees=False, **kargs):
    return get_train_dev_test(
        lambda text_field, label_field, **ka: datasets.SST.splits(text_field, label_field, fine_grained=fine_grained,
                                                                  train_subtrees=train_subtrees),
        **kargs)


def headerless_tsv(data_dir, **kargs):
    return get_train_dev_test(
        lambda text_field, label_field, **ka: mydatasets.PreSplitHeaderlessTsvDataset.splits(text_field, label_field,
                                                                                             data_dir=data_dir),
        **kargs)


def load_word_vectors(vocab):
    word2vec = Magnitude(MagnitudeUtils.download_model("word2vec/GoogleNews-vectors-negative300"))
    emb = np.zeros((len(vocab.itos), word2vec.dim))
    for i, word in tqdm(enumerate(vocab.itos), total=emb.shape[0]):
        emb[i, :] = word2vec.query(word)
    return emb


def print_time(text, timeMeter):
    print("\nTime for {}: {:.2f}{}".format(text, timeMeter.value(), timeMeter.unit))
