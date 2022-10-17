---
layout: post
title: Scalable ML with Spark
categories: [big-data]
tags: [ML]
mathjax: true
show_meta: true
published: true
---

[Epinions.com](https://en.wikipedia.org/wiki/Epinions) was an online marketplace where users posted customer reviews for the goods. When reading each other's comments, they could trust each other. The resulting social network can be represented as a graph where each node represents a user and an oriented edge represents trust. We would like to create a recommendation system on this graph. For each user (node), we want to recommend 10 other users (nodes) to trust.

We are going to train a model that can be used to estimate the probability of an edge in the graph. We will take the 10 most probable edges for the current node that are not presented in the graph, and use them as our recommendations. One of the methods to estimate the quality of recommendations is *Mean Average Precision (MAP)*.

Your task is:

* Implement and train the learning model
* Select 100 nodes from the graph and recommend the top 10 edges.
* Measure recommendation quality using *MAP*
* Make a report

Implementation should be done with Spark using Scala. Estimation of MAP can be done in Python.


## User Recommendation Model

### Motivation

There is a diversity of problems on graphs that one might want to solve. These include

- Graph classification
- Node classification
- Edge classification
- Link prediction
- etc.

The tasks that were listed above can be addressed with the help of machine learning. As you know, the quality of machine learning algorithms highly depends on the input features. It is non-trivial to handcraft these features in the case of graphs and it is tempting to learn useful features using the back-propagation.

From the course of machine learning, you remember the word2vec algorithm. Word2vec is an autoencoder model with a single hidden layer that takes a one-hot representation of a word, encodes it into a dense representation, and then decodes it into the neighborhood of this word. The same approach can be used for predicting the neighborhood of a node in a graph: given one-hot representation of a node, encode it into a dense representation, and use dense representation to predict neighbors of this node.

In the case of the recommendation system, we have a graph with users and items to recommend. We have partially observed some edges in this graph, and we try predicting the presence of an edge between an arbitrary pair (user, item). We hope that features learned during the training process will capture sufficient information about nodes to help us with recommendations.

We are going to use the same autoencoder model for recommending users in the social network of epinions.com.

### Negative Sampling Objective

Let $$\mathcal{G}=(V,E)$$ be a given graph represented by the set of nodes $$V$$ and directed edges $$E$$. Given $$E$$, define the set of neighbors for every node $$N(u) \subset V$$, where $$u$$ is an index of a node in $$V$$.  The idea of negative sampling is simple: train a model to maximize the likelihood of existing edges over random edges.

$$
\begin{aligned}
J &=  \sum_{u\in V} \sum_{n_i\in N(u)} \left( \log  p \left(n_i \vert u\right) - \sum_{k \sim G(u)} log\ p(k \vert u) \right)
\end{aligned}
$$
<!-- $$
\begin{aligned}
J &= \log \sigma(target_{n_i}^T \cdot source_{u}) - \sum_{k \sim G(u)} \log \sigma(target_{k}^T \cdot source_{u})
\end{aligned}
$$ -->

where $$k$$ is an index of a node selected randomly from $$V$$ (uniformly sampled). In this expression, a pair $$(u, n_i)$$ represents a positive example (edge exists), and  $$(u, k)$$ represent a negative example (edge does not exist). This way, we maximize the probability of positive edges, and minimize the probability of negative edges. Due to random sampling, $$k$$ can be selected in such a way that the edge $$(u, k)$$ actually exists. We assume that the probability of such an event is insignificant and the effect can be neglected. The probability $$p(n, u)$$ is parametrized as

$$
p(n, u) = \sigma(target^T_n \cdot source_u)
$$

where $$\sigma$$ is a sigmoid function, $$source_u$$ is a representation of a node $$u$$ as a source node, and $$target_u$$ is a representation of a node $$u$$ as a target node. Thus, each node has two representation vectors associated with it.

### Optimization

It can be shown that minimizing binary cross-entropy is equivalent to maximizing the likelihood (maximizing likelihood is our objective). Thus, we select cross-entropy as the cost function. Cross entropy is defined as

$$
BCE = - y \log p(\hat{y}) - (1-y) \log p(1-\hat{y})
$$

We want to predict the existence of edges between two nodes. We use equations for estimating conditional node probability from the section above and rewrite binary cross-entropy loss as a function of trainable parameters

$$
J = \sum_{(u, n)\in E} \left [ - \log \sigma(target_{n}^T \cdot source_{u}) - \sum_{k \sim G(u)} \log (1 - \sigma(target_{k}^T \cdot source_{u})) \right ]
$$

where $$y=1$$ for positive examples, and $$y=0$$ for negative.

It is recommended to format the training data into the following format
```
source    target    label
1         2         1        // positive example
1         10        0        // negative example
1         30        0        // negative example
2         3         1        // positive example
2         1         0        // negative example
2         4         0        // negative example
```

where only edges (1,2) and (2,3) exist in the graph, and the rest are negatively sampled. This way it is easy to organize the training into batches.

Gradient updates for both parameters $$source$$ and $$target$$ are done with as 
$$
source = source - \alpha \frac{dJ}{d\ source}\\
target = target - \alpha \frac{dJ}{d\ target}
$$

where $$\alpha$$ is the learning rate, $$source$$ is an embedding matrix for source representations, and $$target$$ is an embedding matrix for target representations. $$source_u$$ is the $$u^{th}$$ row of an embedding matrix. For a pair of nodes $$(u, k)$$ the gradients can be computed as

$$
\frac{dJ}{d\ target_k} = source_u \cdot (\sigma(target_{k}^T \cdot source_{u}) - y_{u, k}) \\
\frac{dJ}{d\ source_u} = target_k \cdot (\sigma(target_{k}^T \cdot source_{u}) - y_{u, k})
$$

where $$y_{u, k} \in \{0,1\}$$ is a label for a pair of nodes $$u$$ and $$k$$. In this project, you will need to implement gradient estimation (with Stochastic Gradient Descent) in a distributed fashion and perform parameter updates. 

In practice, you will do Stochastic Gradient Descend in batches. For a single batch, the loss is usually averaged as 

$$
J_B = - \frac{1}{ \vert B \vert  \cdot ( \vert K \vert  +1)}\sum_{(u, n)\in B} \left [ \log \sigma(target_{n}^T \cdot source_{u}) + \sum_{k \sim G(u)} \log \left( 1- \sigma(target_{k}^T \cdot source_{u}) \right) \right ]
$$

where $$ \vert B \vert $$ is the size of a single batch, and $$ \vert K \vert $$ is the number of negative samples. Averaging the loss value for a batch will result in proper gradient scaling.


## MAP
[MAP](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) is a metric widely applied for measuring ranking quality. The notion of relevance is often used to estimate the quality of the ranker. If a document is useful for answering the query, it is called relevant. If it has no meaningful connection to the query - the document is irrelevant.

The search engine returns the ordered list of documents, sorted by descending relevance. We want the top of the list to contain all of the relevant documents for our query.

Assume that your ranker retrieves a list of documents of length $$N_l$$. MAP is the average of a metric called Average Precision over some set of queries 

$$
MAP = \frac{1}{ \vert Q \vert } \sum_{q \in Q} AP(q)
$$

The Average Precision is calculated as

$$
AP = \frac{1}{N_{rel}}\sum_{k=1}^{N_{l}} P(k) \cdot \text{rel}(k)
$$

where $$N_{rel}$$ is the number of the relevant document returned by the ranker, $$P(k)$$ is the precision calculated at the cut-off positions from 1 to k (i.e. relevant documents among the first $$k$$ documents), $$\text{rel}(k)$$ is an indicator function that is equal to 1 when document at position $$k$$ is a relevant document, and 0 otherwise.


## Execution instructions and the dataset

It is recommended to test your software locally, but execute the final version on the cluster. Use your new team credentials to access the cluster.

The data for this task reside in `/epin` folder in the root of HDFS. Use it to train your model. The test file should be used to estimate the MAP score. The data is stored in CSV format. Each line contains a definition of a directed edge in the graph.
```
source_node,destination_node
0,4
0,7
0,9
0,11
0,22
...
```

The model could take a long time to converge. Since you are limited in time and computational resources, we do not require you to train your model until convergence. However,  training process dynamics should be analyzed in the report (loss should decrease).

## Spark Implementation

This section describes the details that might be useful for the Spark implementation of your project. The objective of the assignment can be divided into two major parts: training and evaluation. During training, you learn vector representation of nodes in the graph. During the evaluation, you should try recommending new edges and evaluate the quality of recommendation with MAP. The overall algorithm for your project can be outlined as

1. Training
    1. Read the data
    2. Batch the data
    3. For every batch
        1. Calculate gradients
        2. Update weights
2. Evaluation
    1. For 100 random nodes in the graph, rate possible neighbors
    2. Sort candidates by predicted rating in the descending order
    3. Filter out nodes with existing edges from the recommendation
    4. Get top 10 recommendations and evaluate them using MAP and test data

The first part must be implemented with Spark. The second part can be implemented in Python. Further, we are going to discuss some stages of the training process. Read an appendix for useful scala one-liners.

### Reading the data

The data is either stored in HDFS or on the local machine. Use this method to read the data
```scala
def read_data(path: String, spark: SparkSession): RDD[(Int, Int)] = {
    spark.read.format("csv")
      // the original data is store in CSV format
      // header: source_node, destination_node
      // here we read the data from CSV and export it as RDD[(Int, Int)],
      // i.e. as RDD of edges
      .option("header", "true")
      // State that the header is present in the file
      .schema(StructType(Array(
        StructField("source_node", IntegerType, false),
        StructField("destination_node", IntegerType, false)
      ))) // Define schema of the input data
      .load(path)
      // Read the file as DataFrame
      .rdd.map(row => (row.getAs[Int](0), row.getAs[Int](1)))
      // Interpret DF as RDD 
  }
```

### Linear Algebra

We recommend using [Breeze](https://github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet) library for linear algebra. 
```scala
import breeze.linalg.{DenseMatrix, DenseVector}
```
It supports slices, matrix multiplication, transpose - everything you need to implement the training. For more details on the usage see the official [cheat-sheet](https://github.com/scalanlp/breeze).


### Prepare Batches

During training, we do not want to calculate gradients for all data to make a single parameter update. For this reason, we employ stochastic gradient descend.

There are two approaches to implementing SGD.
- Prepare batches before the training begins. The most obvious way to create batches is to apply a filter (`RDD.filter`) operation and get the first B elements, then second B elements, etc. (B is the size of a batch). This way we will prepare all the batches before the training begins. This will still result in D/B filtering operations (D is the size of the data)
- The second approach includes filtering B random elements on every iteration during training. The basic idea is to call `RDD.filter(rand < 0.1)`, where `rand` is a random number from a uniform distribution. The statement above will keep 10% of the dataset, and this will be our random batch.


### Storing Model Parameters

As in the classical word2vec algorithm, we have two weight matrices of size `(embedding_size, number_of_nodes)`. In Scala, you can store these weights as two DenceMatrix objects (check appendix for how to initialize these matrices). The dimensionality should be `(emb_dim, total_nodes)` for performance reasons.

```scala
val emb_src = create_embedding_matrix(emb_dim, total_nodes)
val emb_tgt = create_embedding_matrix(emb_dim, total_nodes)
```
This will create two objects in the driver's memory. During gradient estimation for a single batch, the value of these matrices will be fixed. For such cases, Spark provides [broadcast objects](https://spark.apache.org/docs/latest/rdd-programming-guide.html#broadcast-variables). These objects are read-only and are distributed efficiently between executors, i.e. every executor will have its copy of a broadcast object.

Before we start the gradient estimation for a batch, we should broadcast weights to all nodes
```scala
var emb_src_broadcast = sc.broadcast(emb_src)
var emb_tgt_broadcast = sc.broadcast(emb_tgt)
```

Then, when we call a map operation, we can retreive the value of the broadcasted object via `value` property
```scala
batch // has type RDD[(Int, Int)], i.e. RDD of edges
 .flatMap( edge => // define lambda
  estimate_gradients_for_edge(
   edge._1, // source node id
   edge._2, // destination node id
   emb_src_broadcast, // snapshot of emb_src
   emb_tgt_broadcast // snapshot of emb_tgt
  ) // has type RDD[(Int, DenseVector[Float])]
 )
```

### Gradients

One possible implementation of function `estimate_gradients_for_edge`:

```scala
def estimate_gradients_for_edge(
                source: Int,
                target: Int,
                emb_src_broad: Broadcast[DenseMatrix[Float]],
                emb_tgt_broad: Broadcast[DenseMatrix[Float]],
               ): Array[((Int, DenseVector[Float]), (Int, DenseVector[Float]))] = {
    val emb_src = emb_src_broad.value // retreive snapshot of emb_src
    val emb_tgt = emb_tgt_broad.value // retreive snapshot of emb_tgt
    
    val src = emb_src(::, source)
    val tgt = emb_tgt(::, target)

    /*
     * Generate negative samples
     * Estimate gradients for positive and negative edges
     */

    // return an array of tuples
    // Tuple((Int, DenseVector), (Int, DenseVector))
    // this tuple contains gradient vectors
    // Array(
    //    ((source, src_grad_positive), (target, tgt_grad_positive)),
    //    ((source, src_grad_negative_1), (random_tgt_1, random_tgt_1_grad)),
    //    ((source, src_grad_negative_2), (random_tgt_2, random_tgt_2_grad)),
    //    ...
    //    ((source, src_grad_negative_K), (random_tgt_K, random_tgt_K_grad))
    // )
  }
```

After applying method `estimate_gradients_for_edge` to every edge in `batch`, we get `RDD[((Int, DenseVector), (Int, DenseVector))]`

First thing that we can do is to separate `src_grads` from `tgt_grads`
```scala
val src_grad_rdd = grads.map(_._1) // RDD[(Int, DenseVector)]
val tgt_grad_rdd = grads.map(_._2) // RDD[(Int, DenseVector)]
```

RDD `src_grad_rdd` contains key-value pairs with gradients, but the keys are not unique. We can aggregate the gradients for a particular node with `reduceByKey` operation. After, we need to download all the distributed gradients to the driver and update parameter matrices
```scala
val in_grads_local = src_grad_rdd // has non unique keys
                        .reduceByKey(_+_) // all keys are uique
                        .collectAsMap() // dowload all the gradients to the driver and convert to HashMap (dictionary)
                        
val tgt_grads_local = tgt_grad_rdd
                        .reduceByKey(_+_) 
                        .collectAsMap()
```

Now that the gradients are collected locally, you need to update the parameter matrices
```scala
for (k <- src_grads_local.keys) {
    /*
     * Scale gradients according to batch size and number of negative samples
     * Update weights in column k of emb_src
     */
}
```

Now, start a new iteration with broadcasting updated `emb_src` and `emb_tgt`.


### Hyperparameters

You can choose parameter values yourself. Here is the set of parameters for the baseline model:

- Number of unique nodes: 40334
- Size of embeddings: 50
- Number of epochs: 20
- Learning rate: 1.0
- Batch size: 10000
- Number of negative samples per positive edge: 20

You are encouraged to try different parameter values to improve the model quality. The only restrictions are
- Top k elements for Mean Average Precision: 10
- Data type: use `float` rather than `double`


## References

- [Dataset](https://snap.stanford.edu/data/soc-Epinions1.html)
<!-- - [node2vec](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf) -->
- [MAP](https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52)

*[MAP]: Mean Average Precision

## Appendix

### Useful scala one-liners
 
Create an array of type `Float` (as opposed to type `Double`) and size 100 containing 0.5 at every position
```scala
Array.fill(100)(0.5f)
```
Create Array initialized to random values using random number generator `rand`
```scala
Array.fill(100)(rand.nextValue())
```
Create `DenseMatrix` from `breeze` library. Use data as initializer (data is an array of size `cols*rows`)
```scala
new DenseMatrix(rows, cols, data)
```
Create empty matrix
```scala
DenseMatrix.zeros[Float](rows, cols)
```
Get number of columns in a DenseMatrix
```
matrix.cols
```
Slice column from `DenseMatrix`
```scala
matrix(::, index) // returns DenseVector[Type]
```
Slice assign to a DenseMatrix
```scala
matrix(::, index) := vector
```
Transpose `DenseVector`
```scala
vector.t
```
Append element to an array
```scala
array :+= element
```
Comprehension
```scala
for (i <- 0 to 4) yield i // creates IndexedSeq(0, 1, 2, 3, 4)
```
Sorting in ascending order by 
```scala
Array[(Int, Float)].sortBy(_._2)
```
Take last k elements
```scala
Array.takeRight(k)
```
Reverse order of a sequence
```
Array.reverse
```
Concatenate into string 
```scala
Array.mkString(",")
```
String interpolation
```scala
s"Epoch $$e, iteration $${i}"
```
Get the first element of a tuple
```
tuple._1
```

### Useful Spark Methods
Create RDD from Iterable 
```scala
sparkContext.parallelize(iterable)
```
Append element index to every element of RDD
```scala
RDD.zipWithIndex() // returns RDD[(OldElement, index)]
```
Repartiotion. Use to reduce memory pressure and increase parallelism
```scala
RDD.repartition(100)
```
Reduce by key, given `RDD[(key, value)]`
```scala
RDD.reduceByKey(_+_) // return RDD[(key, reducedValue)]
```
Collect `RDD[(key, value)]` with unique keys in the driver memory as a `HashMap` (dictionary)
```scala
RDD.collectAsMap()
```
Import random generator
```scala
import org.apache.spark.mllib.random.UniformGenerator
```