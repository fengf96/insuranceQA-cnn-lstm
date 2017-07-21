InsuranceQA using CNN and LSTM
-------
Originally forked from here https://github.com/white127/insuranceQA-cnn-lstm

* Fixed some minor bugs, remove extra code from author's original code.
* Upgrated with tensorflow 1.2 (>= 1.0)
* The pythonic dataset originally comes from https://github.com/codekansas/insurance_qa_python

Before running code, you need to convert the original dataset to author's proposed format
```
cd insurance_qa_python
python3 generate_dataset_for_insuranceQA.py
```

To Run the code of CNN on tensorflow, please install Tensorflow 1.0, and then
```
cd ../../insuranceQA-cnn-lstm
PYTHONPATH=. python3 cnn/tensorflow/insqa_train.py
```

To Run the code of LSTM-CNN (this code is in "cleaner" branch) on tensorflow, please install Tensorflow 1.0, and then
```
cd ../../insuranceQA-cnn-lstm
PYTHONPATH=. python3 lstm_cnn/tensorflow/insqa_train.py
```

My Accuracy:

| Tool          | Method        | Top-1 Accuracy  |
| ------------- |:-------------:| ---------------:|
| Tensorflow    | CNN           |    0.58         |
| Theano        | CNN           |    -            |
| Tensorflow    | LSTM-CNN      |    -            |
| Theano        | LSTM-CNN      |    -            |

解释一下为什么代码和原作者跑出来的不一样，有一个很大的原因是因为数据的negative sample是随机产生的，很容易产生太多毫无关系的负样本。
只有负样本和正样本够接近才利于模型学到pattern，而如果负样本太过随机那模型的准确率也会时高时低。

一个解决这个问题的方法是用tf-idf来产生candidates，这也是insuranceQA的原作者在V2使用的方法https://github.com/shuzi/insuranceQA


-------------from Orignal Author-----------------------------------

See theano and tensorflow folder

This is a CNN/LSTM model for Q&A(Question and Answering), include theano and tensorflow code implementation

theano和tensorflow的网络结构都是一致的:
word embedings + CNN + max pooling + cosine similarity

目前再insuranceQA的test1数据集上，top-1准确率可以达到62%左右，跟论文上是一致的。

这里只提供了CNN的代码，后面我测试了LSTM和LSTM+CNN的方法，LSTM+CNN的方法比单纯使用CNN或LSTM效果还要更好一些，在test1上的准确率可以再提示5%-6%

LSTM+CNN的方法在insuranceQA的test1上的准确率为68%