forked from here https://github.com/white127/insuranceQA-cnn-lstm
fixed some minor bugs, remove extra code from author's original code.
upgrated with tensorflow 1.0
put the original dataset here and convert it to author's proposed format
Run python3 generate_dataset_for_insuranceQA.py
the pythonic dataset is coming from https://github.com/codekansas/insurance_qa_python
--------------------------------


See theano and tensorflow folder

This is a CNN/LSTM model for Q&A(Question and Answering), include theano and tensorflow code implementation

theano和tensorflow的网络结构都是一致的:
word embedings + CNN + max pooling + cosine similarity

目前再insuranceQA的test1数据集上，top-1准确率可以达到62%左右，跟论文上是一致的。

这里只提供了CNN的代码，后面我测试了LSTM和LSTM+CNN的方法，LSTM+CNN的方法比单纯使用CNN或LSTM效果还要更好一些，在test1上的准确率可以再提示5%-6%

LSTM+CNN的方法在insuranceQA的test1上的准确率为68%
