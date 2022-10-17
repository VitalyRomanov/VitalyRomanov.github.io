---
layout: post
title: "Functional Rule Based Models"
categories: [machine-learning]
date: "2019-05-03"
description: ""
tags:
  - Regression
mathjax: true
published: true
---


Sometimes data can be complex and hard to fit using a simple model. However, we can make a model more complex with a set of rules. Consider the data of th efollowing form. It is problematic to describe the data well using only a linear model, however, we can potentially three separate linear models that will descrive data well.

![](https://i.imgur.com/BEqLFO2.png =350x)

The question is how to make this process more data driven.

## Data Clusterization

The data on the example above present two dimensional points. We can start by applying fuzzy clustering to the data, giving way to the emergence of information granules.

![](https://i.imgur.com/Eq5Zis8.png =350x)

You can see a schematic depiction of clusters of information, however, if we apply FCM (Fuzzy C Means clustering), each datapoint belongs to each of the clusters with some level of membership describes as

$$ 
u_{ik} = \frac{1}{ \sum_{j=1}^{ \vert C \vert } \left( \frac{\|[x_k, y_k] - [v_i, w_i]\|}{\|[x_k, y_k] - [v_j, w_j]\|} \right ) ^ {\frac{2}{m-1}} }
$$

$$v_i$$ is the prototype of the cluster in the input dimension, and $$w_i$$ is the prototype of the cluster in the output dimension, and $$[v_i, w_i]$$ is the prototype in both dimensions obtained by concatenating $$v_i$$ and $$w_i$$.

## Local Fit

Now, every cluster can be used to fit a separate model. This can be interpreted as a set of rules in the form

- if $$x_k$$ in $$C_i$$ then $$y_k = f(x, a_i)$$ with certainty $$A_i(x)$$

where $$i=1.. \vert C \vert $$. The degree of membership of $$x$$ to $$C_i$$ (in FCM) is characterized by the membership matrix $$[u_{ik}]$$ (each element equivalent to $$A_i(x_k)$$ in the rule above). If $$f$$ is a regular mapping (a opposed to its granular counterpart), one can assign the degree of certainty to the value $$y$$. This degree is simply 
$$ 
u_{ik} = \frac{1}{ \sum_{j=1}^{ \vert C \vert } \left( \frac{\|x_k - v_i\|}{\|x_k - v_j\|} \right ) ^ {\frac{2}{m-1}} }
$$

Keeping in mind that the points belong to the cluster with different degree of membership, we need to fit several linear models for every rule. The value of linear regression given the set of clusters (rules) is
$$
\hat y = \sum_{i=1}^{ \vert C \vert } A_i(x)\ \mathbf{a_i}^T \mathbf{x}
$$

where $$\mathbf{a_i}$$ is the vector of features, x is the argument value, and $$A_i(x)$$ is the certainty that rule $$i$$ applies to the data $$x$$.

Define 
$$
z_i(x) = A_i(x) x\\
\mathbf{a} = [\mathbf{a_1}\ \mathbf{a_2}\ ...\ \mathbf{a_{ \vert C \vert }}]^T\\
\mathbf{f}(x) = [z_1(x)\ z_2(x)\ ...\ z_{ \vert C \vert }(x)]^T\\
\mathbf{F} = [\mathbf{f}(x_1),\ \mathbf{f}(x_2),\ ...\ \mathbf{f}(x_N)]^T
$$
then 
$$
\hat y = \mathbf{f}^T(x_1) \mathbf{a}
$$
and 
$$
\hat{\mathbf{y}} = \mathbf{Fa}
$$

The optimal solution for $$\mathbf{a}$$ is
$$
\mathbf{a}^{(opt)} = (\mathbf{F}^T \mathbf{F})^{-1} \mathbf{F}^T \mathbf{y}
$$

A solution that is easier to comprehend involves fitting parameters $$\mathbf{a_i}$$ individually for each cluster
$$
\mathbf{a}_i^{(opt)} = (\mathbf{z}_i^T \mathbf{z}_i)^{-1} \mathbf{z}_i^T \mathbf{y}
$$
where $$\mathbf{z}_i = [z_i(x_1), z_i(x_2), ..., z_N(x_N)]^T$$

## Degranularization

The reconstruction of the final value is simply done as a weighted sum of locally fit model

$$
\hat y = \sum_{i=1}^{ \vert C \vert } A_i(x)\ \mathbf{a_i}^T \mathbf{x}
$$ 
