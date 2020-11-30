# 提升方法 (Boosting) {#boosting}

> Breiman called **AdaBoost** the **‘best off-the-shelf classifier in the world’** (NIPS Workshop 1996).

> On the data science competition platform Kaggle, among **29** challenge winning solutions in 2015, **17** used **XGBoost**, a boosting algorithm introduced by Chen and Guestrin.

AdaBoost是一种迭代算法，其核心思想是训练不同的分类器(弱分类器$T$)，然后把这些弱分类器线性组合起来，构成一个更强的最终分类器（强分类器$C$）。

该算法是一个简单的弱分类算法提升过程，这个过程通过不断的训练，可以提高对数据的分类能力。整个过程如下所示：

1. 通过对训练样本$(\mathcal{D},\mathbb{\omega})$的学习得到第$m-1$个弱分类器`WeakClassifier m-1`, $T^{(m-1)}$；

2. 计算得出其分类错误率$\epsilon^{(m-1)}$，以此计算出其弱分类器权重$\alpha^{(m-1)}$与数据权重$\omega^{(m-1)}_i$;

3. 用权重为$\omega^{(m-1)}_i$的数据集训练得到训练弱分类器`WeakClassifier m`, $T^{(m)}$;

4. 重复以上不断迭代的过程;

5. 最终结果通过加权投票表决的方法，让所有弱分类器$T^{(m)}$进行权重为$\alpha^{(m)}$的投票表决的方法得到最终预测输出。

<img src="./plots/4/overview.png" width="60%" style="display: block; margin: auto;" />

- AdaBoost: Schapire and Freund (1997, 2012)

- LogitBoost: Friedman, Hastie, Tibshirani (1998)

- AdaBoost.M1: Schapire and Freund (1996, 1997)

- SAMME: Zhu, Zou, Rosset et al. (2006)

- SAMME.R: Zhu, Zou, Rosset et al. (2006)


## AdaBoost (0,1)

$Y\in\{0,1\}$

1. 初始权重 $\omega^{(0)}_i=\frac{1}{n}$.

> 对于 $m=1,\ldots,M$, 重复以下2-5:
  
>  2. 使用$(\mathcal{D},\mathbf{\omega}^{(m-1)})$，训练弱学习机$T^{(m-1)}$. 
  
>  3. 计算加权分类错误 $\epsilon^{(m-1)}=\sum_{i=1}^n\omega^{(m-1)}_i \mathbb{I}(y_i \neq T^{(m-1)}(\mathbf{x}_i))$.
  
>  4. 计算模型权重 $\alpha^{(m-1)}=\ln\beta^{(m-1)}$, 其中$\beta^{(m-1)}=\frac{1-\epsilon^{(m-1)}}{\epsilon^{(m-1)}}$.
  
>  5. 计算样本权重$\omega^{(m)}_i=\omega^{(m-1)}_i\exp\left( \alpha^{(m-1)}\mathbb{I}(y_i \neq T^{(m-1)}(\mathbf{x}_i)) \right)/w^{(m)}$, 其中$w^{(m)}$为标准化常数。
  
6. 最终预测结果为模型的权重之和较大的那个分类，即  $$C(\mathbf{x})= \underset{k}{\arg \max} \sum_{m=1}^M\alpha^{(m)}\mathbb{I}(T^{(m)}(\mathbf{x})=k)$$。

另外一种等价算法

$Y\in\{-1,1\}$

- 初始权重 $D_1(i)=\frac{1}{n}$.

- 使用$(\mathcal{D},D_m)$，训练弱学习机$h_m$. 

- 计算加权分类错误$\epsilon_m=D_m(i)\mathbf{I}(Y_i\neq h_m(\mathbf{x}_i))$.

- 计算模型权重$\alpha_m=\frac{1}{2}\ln\beta_m$, 其中$\beta_m=\frac{\epsilon_m}{1-\epsilon_m}$.

- 计算样本权重$D_{m+1}(i)=\frac{D_m(i)}{Z_m}\exp\left(-\alpha_mY_ih_m(\mathbf{x_i})\right)$, 其中$Z_m$为标准化常数。

- 最终预测结果为$H(\mathbf{x})= \text{sign}\left(\sum_{m=1}^M \alpha_mh_m(\mathbf{x}) \right)$。

## Logit Boost (real, discrete, gentle AdaBoost)

$Y\in\{-1,1\}$

- 初始弱学习机 $H_0(\mathbf)=h_0(\mathbf{x})=0$.

- 计算预测概率 $p_m(Y_i|\mathbf{x_i})=\frac{1}{1+\exp(-Y_ih_{m-1}(\mathbf{x_i}))}$。注：$p_m(Y_i=1|\mathbf{x_i})+p_m(Y_i=-1|\mathbf{x_i})=1$

- 计算样本权重 $D_m(i)=p_m(Y_i=y_i|\mathbf{x_i})(1-p_m(Y_i=y_i|\mathbf{x_i}))$. 

- 计算工作因变量 $Z_m(i) = y_i(1+\exp(-y_i H_{m-1}(\mathbf{x_i})))$.

- 训练弱学习机$h_m$，使之最小化如下损失函数 $$\sum_{i=1}^N D_m(i)(h_m(\mathbf{x_i})-Z_m(i))^2$$

- 令$H_m=H_{m-1}+h_m$

- 最终预测结果为$\Pr(Y=y|\mathbf{x})= \frac{1}{1+\exp(-yH_M(\mathbf{x_i}))}$, 其中$H_M=h_0+\ldots+h_M$。


## AdaBoost.M1

$Y\in\{1,\ldots,k\}$

- 初始权重 $D_1(i)=\frac{1}{n}$.

- 使用$(\mathcal{D},D_m)$，训练弱学习机$h_m$. 

- 计算加权分类错误$\epsilon_m=D_m(i)\mathbf{I}(Y_i \neq h_m(\mathbf{x}_i)|$.

- 计算模型权重$\alpha_m=-\ln\beta_m$, 其中$\beta_m=\frac{\epsilon_m}{1-\epsilon_m}$.

- 计算样本权重$D_{m+1}(i)=\frac{D_m(i)}{Z_m}\beta_m^{1-\mathbf{I}(Y_i \neq h_m(\mathbf{x}_i))}$, 其中$Z_m$为标准化常数。

- 最终预测结果为 $H(\mathbf{x})= \underset{y\in\{0,\ldots,k\}}{\arg \max} \sum_{m:h_m(\mathbf{x})=y}\alpha_m $

## SAMME (Stage-wise Additive Modeling using a Multi-class Exponential loss function)

$Y\in \{1,\ldots,k\}$

- 初始权重 $D_1(i)=\frac{1}{n}$.

- 使用$(\mathcal{D},D_m)$，训练弱学习机$h_m$. 

- 计算加权分类错误$\epsilon_m=D_m(i)\mathbf{I}(Y_i \neq h_m(\mathbf{x}_i)|$.

- 计算模型权重$\alpha_m=\eta\left(\ln\beta_m + \ln(k-1) \right)$, 其中$\beta_m=\frac{\epsilon_m}{1-\epsilon_m}$.

- 计算样本权重$D_{m+1}(i)=\frac{D_m(i)}{Z_m}\exp\left(\alpha_m\mathbf{I}(Y_i \neq h_m(\mathbf{x}_i))\right)$, 其中$Z_m$为标准化常数。

- 最终预测结果为 $H(\mathbf{x})= \underset{y\in\{0,\ldots,k\}}{\arg \max} \sum_{m:h_m(\mathbf{x})=y}\alpha_m $

## SAMME.R (multi-class real AdaBoost)







## Gradient Boosting

## Newton Boosting

## XGBoost

## Gradient Boost

## XGBoost

<img src="./plots/4/summary.png" width="60%"  style="display: block; margin: auto;" />

## Case study

### Commonly used Python code (for py-beginners)

