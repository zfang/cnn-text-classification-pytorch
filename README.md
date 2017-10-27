## Introduction
This is the implementation of Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch.

1. Kim's implementation of the model in Theano:
[https://github.com/yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)
2. Denny Britz has an implementation in Tensorflow:
[https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
3. Alexander Rakhlin's implementation in Keras;
[https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras)

## Requirement
* python 3
* pytorch > 0.1
* torchtext > 0.1
* numpy

## Usage
```
./main.py -h
```
or 

```
python3 main.py -h
```

You will get:

```
CNN text classificer

optional arguments:
  -h, --help            show this help message and exit
  -lr LR                initial learning rate [default: 0.001]
  -epochs EPOCHS        number of epochs for train [default: 25]
  -batch-size BATCH_SIZE
                        batch size for training [default: 64]
  -log-interval LOG_INTERVAL
                        how many steps to wait before logging training status
                        [default: 1]
  -save-interval SAVE_INTERVAL
                        how many steps to wait before saving [default:0]
  -save-dir SAVE_DIR    where to save the snapshot
  -shuffle              shuffle the data every epoch
  -dropout DROPOUT      the probability for dropout [default: 0.5]
  -max-norm MAX_NORM    l2 constraint of parameters [default: 3.0]
  -kernel-num KERNEL_NUM
                        number of each kind of kernel [default: 100]
  -kernel-sizes KERNEL_SIZES
                        comma-separated kernel size to use for convolution
  -static               fix the embedding
  -device DEVICE        device to use for iterate data, -1 mean cpu [default:
                        -1]
  -no-cuda              disable the gpu
  -snapshot SNAPSHOT    filename of model snapshot [default: None]
  -predict PREDICT      predict the sentence given
  -predictfile PREDICTFILE
                        predict sentences in a file
  -test                 train or test
  -dataset DATASET      specify dataset: sst | mr | none
  -fine-grained         use 5-class sst
  -train-subtrees       train sst subtrees
  -load-word-vectors LOAD_WORD_VECTORS
                        load pre-trained word vectors in binary format
  -load-saved-word-vectors LOAD_SAVED_WORD_VECTORS
                        load saved word vectors
  -debug                debug mode
```

## Train
```
./main.py
```
You will get:

```
Batch[100] - loss: 0.655424  acc: 59.3750%
Evaluation - loss: 0.672396  acc: 57.6923%(615/1066) 
```

## Test
If you has construct you test set, you make testing like:

```
/main.py -test -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt"
```
The snapshot option means where your model load from. If you don't assign it, the model will start from scratch.

## Predict
* **Example1**

	```
	./main.py -predict="Hello my dear , I love you so much ." \
	          -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt" 
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt]...
	
	[Text]  Hello my dear , I love you so much .
	[Label] positive
	```
* **Example2**

	```
	./main.py -predict="You just make me so sad and I have to leave you ."\
	          -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt" 
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt]...
	
	[Text]  You just make me so sad and I have to leave you .
	[Label] negative
	```

## Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

