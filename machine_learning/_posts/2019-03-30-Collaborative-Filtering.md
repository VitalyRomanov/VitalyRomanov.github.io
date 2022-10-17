---
layout: post
title: "Collaborative Filtering"
categories: [machine-learning]
date: "2019-03-30"
description: ""
tags:
  - Recommender
mathjax: true
published: true
---


Consider the following problem. You develop some service that provides some content to a user. This could be music, movies, news, fun facts, etc. You want to do your job in the best possible way and always find the content that would interest your users the most. 

After your users consume content they are asked to rate it. Sometimes users choose not to rate. Thus, you end up with a partial observation of users' preferences. At this point, what you have is the rating matrix $$R \in \mathcal{R}^{ \vert U \vert\times \vert I \vert}$$, where $$U$$ is the set of users, and $$I$$ is the set of items. $$r_{ui}$$ is the rating that user $$u$$ gave to an item $$i$$. You need to find a way to impute the ratings for the items that a user did not grade explicitly.

## Dataset

We are going to use a modified [MovieLens-20m](https://grouplens.org/datasets/movielens/20m/) dataset for training our recommendation system. Originally it contains 20 million ratings. We took the first million ratings and filtered users and movies that rated or were rated less than 20 times. Eventually, we ended up with unique 6687 users and 5064 movies.

## Method

### Matrix Factorization

The idea behind this method is to find such factorization of the rating matrix so that the observed ratings are reconstructed accurately. 

The objective funtion that we want to minimize is 
$$
\underset{P,Q}{min} \sum_{u,i} (r_{u,i} - \langle \mathbf{p}_u,\mathbf{q}_i\rangle)^2 + \lambda (||P||^2 + ||Q||^2) 
$$

where $$\langle \cdot, \cdot \rangle$$ defines inner product between $$u^{th}$$ row of matrix $$P\in\mathcal{R}^{\vert U \vert\times k}$$ and $$i^{th}$$ row of matrix $$Q\in\mathcal{R}^{\vert I \vert\times k}$$. Matrix $$P$$ corresponds to some representation of users and $$Q$$ - to representation of items. Here $$k$$ is the decomposition rank. In other words, $$k$$ is the dimensionality of the representations for users and items space.

The common way to find suitable $$P$$ and $$Q$$ is coordinate descend with ALS. This corresponds to the following procedure:
- find the gradient of the objective with respect to $$P$$, $$Q$$ assumed to be fixed
- adjust $$P$$
- find the gradient of the objective with respect to $$Q$$, $$P$$ assumed to be fixed
- adjust $$Q$$
- repeat until convergence

The calculaiton of the gradients is straightforward, and the update rule is 
$$
\mathbf{p}_u = \mathbf{p}_u - \alpha (\lambda \mathbf{p}_u + \sum_{i \in R(u)} \mathbf{q}_i (\langle \mathbf{p}_u,\mathbf{q}_i\rangle) - r_{u,i}) \\
\mathbf{q}_i = \mathbf{q}_i - \alpha (\lambda \mathbf{q}_i + \sum_{u \in R(i)} \mathbf{p}_u (\langle \mathbf{p}_u,\mathbf{q}_i\rangle) - r_{u,i})
$$

where $$\alpha$$ is the learning rate, $$R(u)$$ is a set of movies graded by user $$u$$, and $$R(i)$$ is a set of users that graded a movie $$i$$.

### Improving with bias

The plain matrix factorization approach is a powerful technique, but it can be further improved by considering additional parameters. For example, some users tend to give low ratings in general, and some items might be widely overvalued by the majority of users. Thus, it is beneficial to add biases for items and users. The objective becomes

$$
\underset{P,Q, \mu, \mathbf{b}^{(U)}, \mathbf{b}^{(I)}}{min} \sum_{u,i} (\langle \mathbf{p}_u,\mathbf{q}_i\rangle + \mu + b_u + b_i - r_{u,i})^2 + \lambda (||P||^2 + ||Q||^2) 
$$

where $$\mu$$ is average item rating over all items, $$\mathbf{b}^{(U)} \in \mathcal{R}^{\vert U\vert}$$ and $$\mathbf{b}^{(I)} \in \mathcal{R}^{\vert I\vert}$$ are bias vectors for users and items correspondingly. 

### [Improving with temporal dependence](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.1951&rep=rep1&type=pdf)

![](https://i.imgur.com/wH1qCbv.png =500x)
*Source: [Collaborative Filtering with Temporal Dynamics](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.1951&rep=rep1&type=pdf)*


There is a strong evidence that biases change over time. Modeling temporal dependence of biases requires significantly more training data and resources. 

$$
\underset{P,Q, \mu, \mathbf{b}^{(U)}, \mathbf{b}^{(I)}}{min} \sum_{u,i} (\langle \mathbf{p}_u(t),\mathbf{q}_i\rangle + \mu + b_u(t) + b_i(t) - r_{u,i}(t))^2 + \lambda (||P(t)||^2 + ||Q||^2) 
$$

we leave this model out of our consideration

## Vectorizing Gradients

It is straightforward to implement gradient update that was provided in the section **Matrix Factorization** when using one of the compiled programming languages. In python, however, such definition of update leads to poor performance due to for loops. For the sake of performance, the update should be translated into vectorized form. Consider the update rule from before. 

Define:
- $$\hat{R} = PQ^T$$ - reconstruction of user ratings
- $$M$$ - indicator mask, $$m_{ui} = 1$$ if user $$u$$ has rated the item $$i$$, and 0 otherwise
- $$\circ$$ - element wise product

Then the gradient update rule becomes
$$
P = P - \alpha (\lambda P - ((P Q^T - R)\circ M)Q) \\
Q = Q - \alpha (\lambda Q - ((P Q^T - R)\circ M)^T P)
$$

Note that this update rule computes all the ratings, although they are not needed for gradient update and are hidden by the mask $$M$$. Although this update rule does more unnecessary calculations, it is still faster than python implementation with loops. For large $$R$$, however, this method is prohibitively memory expensive.

The gradient update rule is provided for the model without additional biases. Vectorizing the bias update should be trivial.

## Sparse matrices

Another way to implement vectorized gradient update is with [sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html). In such matrices, we avoid storing elements equal to 0, and thus, they are much more memory efficient. The trick is to correctly utilize different types of sparse matrices ([see Sparse Matrix Classes](https://docs.scipy.org/doc/scipy/reference/sparse.html#sparse-matrix-classes)) for efficient computation. 

One of useful sparse matrix representations for our case - sparse matrix on coordinate format `coo_matrix`. Coordinate format of $$R$$ is instantiated as
```python
R = coo_matrix((ratings, (userIds, itemIds)), shape=(n_users, n_items))
```
If you used `pandas` for loading the data, `ratigns`, `userIds` and `itemIds` can be obtained as 
```python
ratings = train['rating'].values
userIds = train['userId'].values
itemIds = train['movieId'].values
```

One of possible implementation can include the following steps
1. [Get locations of non-zero elements](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.nonzero.html#scipy.sparse.dok_matrix.nonzero) of matrix $$R$$. Method `nonzero` returns two arrays: `userIds` and `itemIds`. The first - coordinates of non-zero entries along the first axis. Second - along second axis. Alternatively, reuse idices used to construct $$R$$ in the first place
2. Slice $$P$$ and $$Q$$ according to returned indices. You will obtain two matrices $$P^{\tau} \in \mathcal{R}^{\tau \times k}$$ and $$Q^{\tau}\in \mathcal{R}^{\tau \times k}$$ where $$\tau$$ is the number of non-zero entries in $$R$$, and $$k$$ is the dimensionality of the representation vectors for users and items. For more information about indexing in `numpy` read [here](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).
```python
P_tau = P[userIds,:]
Q_tau = Q[itemIds,:]
```
3. Find inner product of $$P^{\tau}$$ and $$Q^{\tau}$$ along the second axis
4. Restore sparse representation of $$\hat{R}$$ using indices obtained on step 1 and inner product obtained on step 3. Look into [`coo_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html).
5. Calculate `((R_hat - R) @ Q)`, update $$P$$. Calculate `((R_hat - R).T @ P)` and update $$Q$$
6. Repeat from step 2 until convergence


You can compare memory consumption of the dense and sparse implementations using [full dataset](https://grouplens.org/datasets/movielens/20m/). 

## Gradient Stability

The original definition of objective sums all the training examples together. This is not very practical for gradient descend optimization with a fixed learning rate. When you sum loss for all training examples, this creates a possibility of very large gradients that remain large even after being multiplied by learning rate. Moreover, when the batch size varies in length, the loss becomes uninterpretable. Averaging loss across all training examples (instead of summing) solves both of these problems. 

Thus, modify the objective, so that the main loss component is Mean Square Error (MSE). Then, you need to adjust the gradients accordingly.

$$
\underset{P,Q, \mu, \mathbf{b}^{(U)}, \mathbf{b}^{(I)}}{min} \frac{1}{|u,i|} \sum_{u,i} (\langle \mathbf{p}_u,\mathbf{q}_i\rangle + \mu + b_u + b_i - r_{u,i})^2 + \lambda (||P||^2 + ||Q||^2) 
$$

where $$\vert u,i\vert $$ is the total number of ratings in the dataset.

Python implementation can be found [on GitHub](https://github.com/VitalyRomanov/collaborative-filtering).

# Additional References

- [Collaborative Filtering](https://web.stanford.edu/~lmackey/papers/cf_slides-pml09.pdf)
- [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)
- [MATRIX FACTORIZATION TECHNIQUES FOR RECOMMENDER SYSTEMS](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- [Collaborative Filtering with Temporal Dynamics](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.1951&rep=rep1&type=pdf)

*[ALS]: Alternating Least Squares
*[CF]: Collaborative Filtering

