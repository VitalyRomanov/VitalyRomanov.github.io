---
layout: post
title: "Fuzzy C Means Clustering"
categories: [machine-learning]
date: "2019-04-27"
description: ""
tags:
  - Clustering
mathjax: true
published: true
---


Clustering is one of the ways to construct information granules. Specifically, the sole goal of clustering is to discover a hidden structure in numerical data in the form of clusters -- information granules. One of the most common algorithms, K-Means Clustering, presents information granules in the form of sets, with a strict membership boundary. An alternative algorithm, Fuzzy C Means Clustering, allows to describe information granules in the form of fuzzy set - a set that is defined with a membership function $$A(x)$$ that assigns a numerical value from the interval [0,1] to every element $$x$$.

## Objective

Given the collection of n-dimensional data sets $$\{x_k\}, k=1,2,...,N$$, the task is to determine a collection of clusters C that minimizes the objective
$$
Q = \sum_{i=1}^{ \vert C \vert } \sum_{k=1}^N u_{ik}^m \|x_k - v_c\|^2
$$

where $$v_1, v_2, ..., v_{ \vert C \vert }$$ are n-dimensional prototypes of the clusters and $$U=[u_{ik}]$$ is a partition matrix that expresses the allocation of the data points to particular clusters. Alternatively, $$u_{ik}$$ can be considered as a membership degree of a fuzzy set specified by the cluster $$i$$. The fuzzification coefficient $$m >1.0$$ expresses the impact of the membership grades on the individual clusters and produces certain geometry of the information granules. The partition matrix is subject to **two constraints**

1. $$0 < \sum_{k=1}^N u_{ik} < N;\ i=1,2,..., \vert C \vert $$
2. $$\sum_{i=1}^{ \vert C \vert } u_{ik} = 1;\ k=1,2,...,N$$

In the case of using euclidean distances, one can derive update equations for minimizing Q using Lagrange multipliers
$$
v_i = \frac{\sum_{k=1}^N u_{ik}^m x_k}{\sum_{k=1}^N u_{ik}^m} \\
u_{ik} = \frac{1}{ \sum_{j=1}^{ \vert C \vert } \left( \frac{\|x_k-v_i\|}{\|x_k - v_j\|} \right ) ^ {\frac{2}{m-1}} }
$$

The updates are performed iteratively according to the algorithm
1. Initialize U
2. Update $$v_1, v_2, ... v_{ \vert C \vert }$$
3. Update U
4. If $$\underset{i,k}{max}\  \vert u_{ik}^{(t+1)} - u_{ik}^{(t)} \vert  < \epsilon$$, otherwise return to step 2

$$\epsilon$$ is a termination criterion in the interval [0,1]. The increase of hyperparameter $$m$$ encourages information granules to be more specific.
