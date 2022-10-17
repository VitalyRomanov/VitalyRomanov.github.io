---
layout: post
title: "Information Theoretic Approach to Classification"
categories: [machine-learning]
date: "2018-10-10"
description: "Logistic regression is the simplest form of classification. We all know that the cost function is the cross entropy loss. But why?"
tags:
  - classification
mathjax: true
---

Consider the task of classification, where you solve a problem of mapping a set of features $$X$$ to a target label $$y$$, so that $$C(X)=y$$, where $$C$$ is your classification function. Now, it is highly possible that your set of features does not provide a perfect explanation of the target class, and thus you may find several data samples that are identical, but have different class labels, i.e. $$X_1=X_2, C(X_1)=y_1, C(X_2)=y_2$$. This is usually called "noisy data".
<!--more-->
The process of training a model condescends to finding a function $$C$$ that classifies correctly, but what is the measure of correctness, especially in the presence of the noise?

## Finding Help from Information Theory

The fact that we cannot certainly tell the correct label for a specific instance of data means that we are dealing with a probability distribution over the set of target labels $$p(y\|X)$$ . Intuitively, we want out classifier to create similar distribution over the estimated label $$\hat{y}$$. The measure of the dissimilarity of two distributions is Kullback-Leibler divergence, which we need to minimize. Thus, the target loss is

$$
L=\sum_{unique\ X \in \mathcal{D}} D_{KL}\left(p(Y|X)||p(\hat{Y}|X)\right)
$$

## Explaining the Math

Let us first consider the term inside the sum $$D_{KL}\left(p(y\|X)\|\|p(\hat{y}\|X)\right)$$.

$$
\begin{aligned}
D_{KL}\left(Y|X||\hat{Y}|X\right) & = \sum_{c \in C} p(Y = c|X) \frac{p(Y = c | X)}{p(\hat{Y} = c |X)} \\
& = -\sum_{c\in C} p(Y = c|X) \log p(\hat{Y} = c|X) \\
& \qquad + \sum_{c\in C} p(Y = c|X) \log p(Y = c |X) \\
& = H(Y|X, \hat{Y}|X) - H(Y|X)
\end{aligned}
$$

where $$C$$ is the set of distinct class labels, the first term in the last line of the equation above is the cross entropy, and the the second term is the conditional entropy of the target label $$y$$ given $$X$$.

As you might have guessed, the classification function participates only in the first term, and the second term is effectively a constant for our dataset. Thus minimization of Kullback-Leibler divergence is reduced to minimization of cross entropy.

## Putting Everything Together

We have discovered that minimization of Kullback-Leibler divergence for finding a classifier is equivalent to minimization of cross entropy. Now lets combine the loss for all the data samples in the dataset​ and obtain the final loss function.

$$
\begin{aligned}
L & = \sum_{unique\ X \in \mathcal{D}} H(Y|X, \hat{Y}|X) \\
& = -\sum_{unique\ X \in \mathcal{D}} \sum_{c\in C} p(Y = c|X) \log p(\hat{Y} = c|X)
\end{aligned}
$$

In the equation above, we do not know the actual probability $$p(y_i\|X)$$, and the best we can do is to estimate this value from our data.

$$
\begin{aligned}
L & = -\sum_{unique\ x \in \mathcal{D}} \sum_{c\in C} p(Y = c|x) \log p(\hat{Y} = c|x) \\
&= -\sum_{unique\ x \in \mathcal{D}} \sum_{c\in C} \frac{\sum_{x_i \in \mathcal{D}}I(x_i=x, y_i = c)}{|\mathcal{D}|} \log p(\hat{Y} = c|x)\\
& = -\sum_{x_i \in \mathcal{D}} \frac{1}{|\mathcal{D}|}\log p(\hat{y_i}|X)
\end{aligned}
$$

Since for every data sample $$x_i \in \mathcal{D}$$  has exactly one class label, we can rearrange the order of summation

$$
\begin{aligned}
L & = - \sum_{x_i \in \mathcal{D}} \sum_{unique\ x \in \mathcal{D}} \sum_{c\in C} \frac{I(x_i=x, y_i = c)}{|\mathcal{D}|} \log p(\hat{Y} = c|x)\\
&= - \sum_{x_i \in \mathcal{D}} \sum_{unique\ x \in \mathcal{D}} \frac{I(x_i=x)}{|\mathcal{D}|} \log p(\hat{Y} = y_i|x)\\
&= - \sum_{x_i \in \mathcal{D}} \frac{1}{|\mathcal{D}|} \log p(\hat{Y} = y_i|x_i)
\end{aligned}
$$


## Expressing From Likelihood

It is a well known fact that minimization of Kullback-Leibler Divergence is equivalent to maximization of likelihood. This can be easily shown by defining what the likelihood is

$$
\begin{aligned}
log(p(x|D)) &= log(\prod_{c \in C} \prod_{j \in I_c} p(\hat{Y}=y_j|x_j)) \\
&= \sum_{c \in C} \sum_{j \in I_c} log( p(\hat{Y}=y_j|x_j))
\end{aligned}
$$

where $$C$$ is the set of distinct classes, and $$I_c$$  is the set of data sample indices with the class label equal to $$c$$. Since a data sample can have only one class label, the sets $$I_c$$ are disjoint and we can bring everything under one summation operator

$$
\begin{aligned}
log(p(x|D)) &= \sum_{c \in C} \sum_{j \in I_c} log( p(\hat{Y}=y_j|x_j)) \\
&= \sum_{x_i \in D}  log(p(\hat{Y}=y_i|x_i))
\end{aligned}
$$

As it is easy to see, maximizing log likelihood is equivalent to minimizing cross entropy. Simple, but not as fun.