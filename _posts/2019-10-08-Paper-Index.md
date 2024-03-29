---
layout: post
title:  "文章汇总"
subtitle: "读过的文章的分类汇总"
author: "yuetong"
header-img: "img/state_of_the_art.jpg"
date:   2019-10-08
tags:
    - 深度学习
    - 分布式训练
    - 文章总结

---

# 文章汇总
在这里，我将近期读过的一些文章做一个汇总与梳理。之后这篇汇总涉及到的篇目将随着阅读的深入不断的增加。
这里包括的文章大多是与分布式深度神经网络训练相关的，不过这个领域包括许多细分的方向，各篇文章侧重的重点有所不同，当然不同方向的文章的关注点也有交集。为了更好的整理这个目录性质的文章汇总，我依据个人理解粗略对文章进行分类，并放入相应的类别下。某些文章如果与多个主题相关，它们将会在多个类别下出现。


## 综述

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis](http://arxiv.org/abs/1802.09941)|arXiv|2018-02|
|[A Hitchhiker’s Guide On Distributed Training of Deep Neural Networks](http://arxiv.org/abs/1810.11787)|arXiv|2018-10|

## Large Batch Training
### 系统整体设计

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
| [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)    |   arXiv |  2017-06  |
| [ImageNet Training in Minutes](https://dl.acm.org/citation.cfm?id=3225069) | ICPP 2018 | 2018-08 |
| [Extremely Large Minibatch SGD: Training ResNet-50 on ImageNet in 15 Minutes](https://arxiv.org/abs/1711.04325) | arXiv | 2017-11 |
| [Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes](https://arxiv.org/abs/1807.11205) | arXiv | 2018-07|
|[Optimizing Network Performance for Distributed DNN Training on GPU Clusters: ImageNet/AlexNet Training in 1.5 Minutes](http://arxiv.org/abs/1902.06855)|arXiv|2019-02|
|[Exascale Deep Learning for Climate Analytics](http://arxiv.org/abs/1810.01993)|SC 19|2018-10|
|Speeding up ImageNet Training on Supercomputers|SysML 2018|2018|
|[Revisiting distributed synchronous SGD](https://arxiv.org/abs/1604.00981)|ICLR 2016|2016-04|
|Ako: Decentralised Deep Learning with Partial Gradient Exchange|SoCC 2016|2016|
|CROSSBOW: Scaling Deep Learning with Small Batch Sizes on Multi-GPU Servers|SysML 2019|2019-08|


### 算法分析与优化

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[Large Batch Training of Convolutional Networks](http://arxiv.org/abs/1708.03888)|arXiv|2017-08|
| [Train longer, generalize better: closing the generalization gap in large batch training of neural networks](http://arxiv.org/abs/1705.08741) | NIPS 2017|2017-05|
|[On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](http://arxiv.org/abs/1609.04836)|ICLR 2017|2016-09|
|[Reducing BERT Pre-Training Time from 3 Days to 76 Minutes](http://arxiv.org/abs/1904.00962)|arXiv|2019-04
|[Don't Decay the Learning Rate, Increase the Batch Size](http://arxiv.org/abs/1711.00489)|ICLR 2018|2017-11|
|[Large Batch Size Training of Neural Networks with Adversarial Training and Second-Order Information](http://arxiv.org/abs/1810.01021)|arXiv|2018-10|
|Augment your batch: better training with larger batches|arXiv|2019-01|

## 通信优化
### Ring AllReduce

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[Horovod: fast and easy distributed deep learning in TensorFlow](http://arxiv.org/abs/1802.05799)|arXiv|2018-02|
|A Distributed Synchronous SGD Algorithm with Global Top-$k$ Sparsification for Low Bandwidth Networks|ICDCS 2019|2019-01|

### Parameter Server

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[Parameter Box: High Performance Parameter Servers for Efficient Distributed Deep Neural Network Training](http://arxiv.org/abs/1801.09805)|arXiv|2018-01|
|[BML: A High-performance, Low-cost Gradient Synchronization Algorithm for DML Training](http://papers.nips.cc/paper/7678-bml-a-high-performance-low-cost-gradient-synchronization-algorithm-for-dml-training.pdf)|NIPS 2018|2018|
|[GeePS: Scalable deep learning on distributed GPUs with a GPU-specialized parameter server](http://doi.acm.org/10.1145/2901318.2901323)|EuroSys 2016|2016-04|
|[Large Scale Distributed Deep Networks](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf)|NIPS 2012|2012|
|[Priority-based Parameter Propagation for Distributed DNN Training](http://arxiv.org/abs/1905.03960)|SysML 2019|2019-05|
|Round-Robin Synchronization: Mitigating Communication Bottlenecks in Parameter Servers|infocom 2019|2019|

### 其他

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[Parallax: Sparsity-aware Data Parallel Training of Deep Neural Networks](http://arxiv.org/abs/1808.02621)|EuroSys 2019|2018-08|
|[Optimization of Collective Reduction Operations](http://link.springer.com/10.1007/978-3-540-24685-5_1)|ICCS 2004|2004-06|
|[MXNET-MPI: Embedding MPI parallelism in Parameter Server Task Model for scaling Deep Learning](http://arxiv.org/abs/1801.03855)|Proceedings of ACM Conference|2018-01|
|[BigDL: A Distributed Deep Learning Framework for Big Data](http://arxiv.org/abs/1804.05839)|arXiv|2018-04|
|Project Adam: Building an Efficient and Scalable Deep Learning Training System|OSDI 2014|2014|
|RedSync: Reducing synchronization bandwidth for distributed deep learning training system|JPDC 2019|2018|
|Ako: Decentralised Deep Learning with Partial Gradient Exchange|SoCC 2016|2016|
|TicTac: Accelerating Distributed Deep Learning with Communication Scheduling|SysML 2019|2019|

## 并行模式
### 数据并行
### 模型并行
### 流水线训练

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[PipeDream: Fast and Efficient Pipeline Parallel DNN Training](http://arxiv.org/abs/1806.03377)|arXiv|2018-06|
|[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](http://arxiv.org/abs/1811.06965)|arXiv|2018-11|

### 混合并行

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
| [One weird trick for parallelizing convolutional neural networks](http://arxiv.org/abs/1404.5997) | arXiv |2014-04 |
|Beyond Data and Model Parallelism for Deep Neural Networks|SysML 2019|2019|

## 通信量化&压缩

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|AdaComp: Adaptive Residual Gradient Compression for Data-Parallel Distributed Training|AAAI 2018|2018|
|[Communication Quantization for Data-Parallel Training of Deep Neural Networks](http://ieeexplore.ieee.org/document/7835789/)|MLHPC|2016-11|
|[Communication Quantization for Data-Parallel Training of Deep Neural Networks](http://ieeexplore.ieee.org/document/7835789/)|MLHPC|2016-11|
|[TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning](http://papers.nips.cc/paper/6749-terngrad-ternary-gradients-to-reduce-communication-in-distributed-deep-learning.pdf)|NIPS 2017|2017|
|RedSync: Reducing synchronization bandwidth for distributed deep learning training system|JPDC 2019|2018|

## 优化算法
### 异步算法

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[Slow and Stale Gradients Can Win the Race: Error-Runtime Trade-offs in Distributed SGD](http://arxiv.org/abs/1803.01113)|AISTATS 2018|2018-03|

### 模型平均

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[Don't Use Large Mini-Batches, Use Local SGD](https://arxiv.org/abs/1808.07217)|arXiv|2018-08|
|[Reducing global reductions in large-scale distributed training](http://dl.acm.org/citation.cfm?doid=3339186.3339203)|ICPP 2019|2019-08|
|[On the convergence properties of a $K$-step averaging stochastic gradient descent algorithm for nonconvex optimization](http://arxiv.org/abs/1708.01012)|arXiv|2017-08|
|[Scalable Distributed DL Training: Batching Communication and Computation](https://ocs.aaai.org/ojs/index.php/AAAI/article/view/4465)|AAAI 2019|2019|
|[Local SGD Converges Fast and Communicates Little](http://arxiv.org/abs/1805.09767)|arXiv|2018-05|
|[Parallel Restarted SGD with Faster Convergence and Less Communication: Demystifying Why Model Averaging Works for Deep Learning](http://arxiv.org/abs/1807.06629)|AAAI 2019|2018-07|
|A[daptive Communication Strategies to Achieve the Best Error-Runtime Trade-off in Local-Update SGD](http://arxiv.org/abs/1810.08313)|SysML 2019|2018-10|
|[Can Decentralized Algorithms OutperformCentralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent](http://papers.nips.cc/paper/7117-can-decentralized-algorithms-outperform-centralized-algorithms-a-case-study-for-decentralized-parallel-stochastic-gradient-descent.pdf)|NIPS 2017|2017|
|[Experiments on Parallel Training of Deep Neural Network using Model Averaging](http://arxiv.org/abs/1507.01239)|arXiv|2015-07|
|[Cooperative SGD: A Unified Framework for the Design and Analysis of Communication-Efficient SGD Algorithms](http://arxiv.org/abs/1808.07576)|arXiv|2018-08|

## 其他

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[Fast Distributed Deep Learning via Worker-adaptive Batch Sizing](http://arxiv.org/abs/1806.02508)|arXiv|2018-06|
|Dynamic Mini-batch SGD for Elastic Distributed Training: Learning in the Limbo of Resources|arXiv|2019-04|
|CROSSBOW: Scaling Deep Learning with Small Batch Sizes on Multi-GPU Servers|SysML 2019|2019-08|
|Parallelized Training of Deep NN – Comparison of Current Concepts and Frameworks|DIDL 2018|2018-12|

## 经典文章
### 网络模型

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)|CVPR 2016|2015-12|
|[Exploring the Limits of Language Modeling](http://arxiv.org/abs/1602.02410)|arXiv|2016-02|
|[ImageNet classification with deep convolutional neural networks](http://dl.acm.org/citation.cfm?doid=3098997.3065386)|NIPS 2012|2012|
|[Communication-Efficient Learning of Deep Networks from Decentralized Data](http://arxiv.org/abs/1602.05629)|AISTATS 2017|2016-02|

### 数据集

|论文名称|发表刊物|发表时间|
| :--------: | :--------:| :------: |
|[One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling](http://arxiv.org/abs/1312.3005)|arXiv|2013-12|
