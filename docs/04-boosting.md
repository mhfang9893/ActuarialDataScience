# 提升方法 (Boosting) {#boosting}

> Breiman called **AdaBoost** the **‘best off-the-shelf classifier in the world’** (NIPS Workshop 1996).

> On the data science competition platform Kaggle, among **29** challenge winning solutions in 2015, **17** used **XGBoost**, a boosting algorithm introduced by Chen and Guestrin.

Boosting是一种迭代算法，其核心思想是针对同一个训练集训练不同的分类器(弱分类器)，然后把这些弱分类器线性组合起来，构成一个更强的最终分类器（强分类器）。

该算法其实是一个简单的弱分类算法提升过程，这个过程通过不断的训练，可以提高对数据的分类能力。整个过程如下所示：

1. 先通过对N个训练样本的学习得到第一个弱分类器WeakClassifier1；

2. 计算得出其分类错误率，以此计算出其弱分类器权重alpha(1)与数据权重W(1)

3. 用权重为W(1)的数据集训练得到训练弱分类器WeakClassifier(2) 

4. 重复以上不断迭代的过程

5. 最终结果通过加权投票表决的方法，让所有弱分类器进行加权投票表决的方法得到最终预测输出，计算最终分类错误率，如果最终错误率低于设定阈值（比如5%），那么迭代结束

<img src="./plots/4/overview.png" width="60%" style="display: block; margin: auto;" />

**符号说明**

- $M$总迭代次数，$m$为第$m$次迭代

- $h_m$为第$m$个弱学习器

- $D_m(i)$为第$m$次迭代后的第$i$个样本的权重

- $\epsilon_m$为第$m$个弱学习的错误率


## AdaBoost

## AdaBoost.M1

## SAMME

## SAMME.R

## Logit Boost

## Gradient Boosting

## Newton Boosting

## XGBoost

## Gradient Boost

## XGBoost

<img src="./plots/4/summary.png" width="60%"  style="display: block; margin: auto;" />

## Case study

### Commonly used Python code (for py-beginners)

