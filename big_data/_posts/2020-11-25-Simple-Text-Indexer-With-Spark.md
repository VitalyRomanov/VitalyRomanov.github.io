---
layout: post
title: Simple Text indexer With Spark
categories: [big-data]
tags: []
mathjax: true
show_meta: true
published: true
---

The goal is to implement a naive text indexer and document retriever. These modules are often found in search engines. Conceptually, search engines were the first who tackled the problem of Big Data with the constraint of low latency response. Imagine an average search engine that has millions of documents in its index. Every second it receives hundreds to thousands of queries and requires to produce a list of the most relevant documents at sub-millisecond speed. 

The problem of finding relevant information is one of the key problems in the field of Information Retrieval, which can be subdivided into two related tasks:

- Document indexing
- Query answering (Document retrieval)

While the second task is more related to low latency processing, the indexing step can be done offline and is more relevant to the idea of batch processing. Systems like Hadoop YARN can be used to create the index of a text corpus that is too large to fit on one machine. For the sake of this assignment, you are going to do a naive implementation of both tasks in Spark using MapReduce paradigm.

The diagram for such a search engine is shown on the figure below.
![](https://www.dropbox.com/s/hkxjcimcx0g4lv0/Search%20Engine%20Flow.png?dl=1)

Before going into the details, let us review a basic approach to information retrieval.


## Basics of Information Retrieval for Text

The most common task in IR is textual information retrieval. Whenever a user submits a query, the ranking engine should compute the set of the most relevant documents in its collection. To complete this task, the engineer should determine the representation format for both documents and queries, and define the measure of relevance of a query for a particular document. One of the most simple IR models is the TF/IDF Vector Space model. 

### Document Representation

To facilitate the understanding of the vector space model, let us define a toy corpus that consists of three documents


| Doc Id | Content |
| -------- | -------- |
| 1     | I wonder how many miles I’ve fallen by this time?     |
| 2     | According to the latest census, the population of Moscow is more than two million.     |
| 3     | It was a warm, bright day at the end of August.     |
| 4     | To be, or not to be? |
| 5     | The population, the population, the population |


To define the vector space model, we need to introduce three concepts: vocabulary, term frequency (TF), and inverse document frequency (IDF).

Term
:    A term is a unique word.

Vocabulary
:    Vocabulary is a set of unique terms present in the corpus. For the example above the vocabulary can be defined as 
```
{'a', 'according', 'at', 'august', 'bright', 'by', 'census', 'day', 'end', 'fallen', 'how', 'i', 'is', 'it', 'i’ve', 'latest', 'many', 'miles', 'million', 'more', 'moscow', 'of', 'population', 'than', 'the', 'this', 'time', 'to', 'two', 'warm', 'was', 'wonder', 'or', 'not', 'be'}
```
For the ease of further description, associate each term in vocabulary with a unique id
```
(0, 'a'), (1, 'according'), (2, 'at'), (3, 'august'), (4, 'bright'), (5, 'by'), (6, 'census'), (7, 'day'), (8, 'end'), (9, 'fallen'), (10, 'how'), (11, 'i'), (12, 'is'), (13, 'it'), (14, 'i’ve'), (15, 'latest'), (16, 'many'), (17, 'miles'), (18, 'million'), (19, 'more'), (20, 'moscow'), (21, 'of'), (22, 'population'), (23, 'than'), (24, 'the'), (25, 'this'), (26, 'time'), (27, 'to'), (28, 'two'), (29, 'warm'), (30, 'was'), (31, 'wonder'), (32, 'or'), (33, 'not'), (34, 'be')
```

Term Frequency
:    Term Frequency (TF) is the frequency of occurrence of a term $$t$$ in a document $$d$$ (how many times the term appears in the document). The previous documents can be represented with TF as follows, given the term format `(id, frequency)`



| Doc Id | Content |
| -------- | -------- |
| 1     | `(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 1), (6, 0), (7, 0), (8, 0), (9, 1), (10, 1), (11, 1), (12, 0), (13, 0), (14, 1), (15, 0), (16, 1), (17, 1), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (24, 0), (25, 1), (26, 1), (27, 0), (28, 0), (29, 0), (30, 0), (31, 1), (32, 0), (33, 0), (34, 0)`   |
| 2     | `(0, 0), (1, 1), (2, 0), (3, 0), (4, 0), (5, 0), (6, 1), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 1), (13, 0), (14, 0), (15, 1), (16, 0), (17, 0), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 2), (25, 0), (26, 0), (27, 1), (28, 1), (29, 0), (30, 0), (31, 0), (32, 0), (33, 0), (34, 0)` |
| 3     | `(0, 1), (1, 0), (2, 1), (3, 1), (4, 1), (5, 0), (6, 0), (7, 1), (8, 1), (9, 0), (10, 0), (11, 0), (12, 0), (13, 1), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 1), (22, 0), (23, 0), (24, 1), (25, 0), (26, 0), (27, 0), (28, 0), (29, 1), (30, 1), (31, 0), (32, 0), (33, 0), (34, 0)`  |
| 4     | `(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (24, 0), (25, 0), (26, 0), (27, 2), (28, 0), (29, 0), (30, 0), (31, 0), (32, 1), (33, 1), (34, 2)` |
| 5     | `(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 3), (23, 0), (24, 3), (25, 0), (26, 0), (27, 0), (28, 0), (29, 0), (30, 0), (31, 0), (32, 0), (33, 0), (34, 0)` |

As you can see, this representation format is very sparse, and we could preserve space if we would store it as a sparse representation (i.e., remove all zero entries)

| Doc Id | Content |
| -------- | -------- |
| 1     | `(5, 1), (9, 1), (10, 1), (11, 1), (14, 1), (16, 1), (17, 1), (25, 1), (26, 1)`   |
| 2     | `(1, 1), (6, 1), (12, 1), (15, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 2), (27, 1), (28, 1)` |
| 3     | `(0, 1), (2, 1), (3, 1), (4, 1), (7, 1), (8, 1), (13, 1), (21, 1), (24, 1), (29, 1), (30, 1)`  |
| 4     | `(27, 2), (32, 1), (33, 1), (34, 2)` |
| 5     | `(22, 3), (24, 3)` |

Inverse Document Frequency
:    IDF shows in how many documents the term has appeared. The measure signifies how common a particular term is. In case IDF is high, the presence of this term in the document does not help us to distinguish between documents. IDFs for our corpus are (in `(id, IDF)` format)

```
(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 2), (22, 2), (23, 1), (24, 3), (25, 1), (26, 1), (27, 2), (28, 1), (29, 1), (30, 1), (31, 1), (32, 1), (33, 1), (34, 1)
```

You can see that the term `be` has occurred twice in the last document, but it's IDF is only `1`. The term `the` occurred in document `2` and `3`; hence its IDF is `2`.

Sometimes IDF is defined as 
$$
IDF(t) = log \frac{N}{n(t)}
$$
where $$N$$ is the overall number of documents in the collection, and $$n(t)$$ is the number of documents containing the term $$t$$.

Other definitions are also possible. You are free to use any of them.

TF/IDF Weights
:    TF/IDF weights are term frequencies normalized by IDF. 

TF/IDF Weights for our documents are

| Doc Id | Content |
| -------- | -------- |
| 1     | `(5, 1), (9, 1), (10, 1), (11, 1), (14, 1), (16, 1), (17, 1), (25, 1), (26, 1)`   |
| 2     | `(1, 1), (6, 1), (12, 1), (15, 1), (18, 1), (19, 1), (20, 1), (21, 0.5), (22, 0.5), (23, 1), (24, 0.66), (27, 0.5), (28, 1)` |
| 3     | `(0, 1), (2, 1), (3, 1), (4, 1), (7, 1), (8, 1), (13, 1), (21, 0.5), (24, 0.33), (29, 1), (30, 1)`  |
| 4     | `(27, 1), (32, 1), (33, 1), (34, 2)` |
| 5     | `(22, 1.5), (24, 1)` |

Here you can notice that terms with ids `{21, 22, 24, 27}` (`of`, `population`, `the`, `to`) are downscaled.

:::info
We have discussed an approach when each word is assigned a unique ID. Alternatively, one can assign ID usign a hash function. This way one can easily control the size of the final vocabulary.
:::

### Vector Space Model for Information Retrieval 

In the basic vector space model, both documents and queries are represented with corresponding vectors, which capture TF/IDF weights of a document and the query. 

The simplest way to convert TF/IDF weights to a vector interpreted by the computer is to index the array with word Ids and record TF/IDF value. In this case, the vector for document `5` will be 

```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

The query `the population` will result in a vector
```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

The function that determines the relevance of a document to a query is the inner product (scalar product) of the two vectors

$$
r(q,d) = \sum_{i=1}^{|V|} q_i \cdot d_i
$$

where $$|V|$$ is the size of the vocabulary, and $$q_i$$ is the TF/IDF weight of the $$i^{th}$$ term in the query.

As you can see, the naive vector space representation with arrays is very sparse. Alternatively, you can create a sparse vector representation. The difference between a sparse vector and a regular vector is that a sparse vector stores only non zero values. Such a vector can be implemented with a dictionary (in scala terms - [HashMap](https://docs.scala-lang.org/overviews/collections/maps.html)).

:::danger
Map and HashMap are not serializable, which presents additional difficulties for using them with Spark. We suggest storing a sparse vector as a sorted list. Then, an intersection of query and the document can be computed [cheaply](https://www.geeksforgeeks.org/union-and-intersection-of-two-sorted-arrays-2/). Another way of handling this problem is passing data as an Array of tuples, but inside the map function you can transform it into a Dictionary by calling `toMap`.
:::

Sparse vector representation for the document `5` is

```
[(22: 1.5), (24: 1)]
```

and for the query

```
[(22: 0.5), (24: 0.33)]
```

The relevance function is easily reformulated as

$$
r(q,d) = \sum_{i: i\in d, i\in q} q_i \cdot d_i
$$

where the summation is performed over all the terms that are present in both the query and the document. From the implementation standpoint, the dictionary structure is more appealing because the keys are unique and form sets. The common terms for a document and a query can be easily found by the set intersection (which is usually very fast).

If we compute the relevance of the query `the population` for our toy corpus, the result would be

```
doc 5: 1.08
doc 2: 0.4678
doc 3: 0.1089
doc 4: 0
doc 1: 0
```

As you can see, even though document `5` contains arguably less information, it is still the first according to our relevance rating.


### BM25 for Information Retrieval

[Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) is widely accepted as a much better ranking function for ad-hoc (no relevance data available) information retrieval. The formula looks like this

$$
r(q,d) = \sum_{i: i\in d, i\in q} IDF(d_i) \frac{d_i \cdot (k_1 + 1)}{d_i + k_1 \cdot (1 - b + b \cdot \frac{\vert d\vert }{avgdl})}
$$

For the sake of simplicity, we can assume the same definition for IDF as above, $$d_i$$ is term frequency, $$avgdl$$ is the average document length in the corpus. $$\vert d\vert $$ is the size of the current document, $$b$$ and $$k_1$$ are free parameters (usually set to 0.75 and 2.0).

This ranking function solves several problems with the vector space model approach, including the bias towards longer documents.

### Assessing Quality with Mean Average Precision

[MAP](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) is a metric widely applied for measuring ranking quality. The notion of relevance is often applied to estimate the quality of the ranker. If a document is useful for answering the query, it is called relevant. If it has no meaningful connection to the query - the document is not relevant. 

The search engine returns the ordered list of documents, sorted by descending relevance. We want the top of the list to contain all of the relevant documents for our query.

Assume that your ranker retrieves a list of documents of length $$N_l$$. MAP is the average of a metric called Average Precision over some set of queries 

$$
MAP = \frac{1}{\vert Q\vert} \sum_{q \in Q} AP(q)
$$

The Average Precision is calculated as

$$
AP = \frac{1}{N_{rel}}\sum_{k=1}^{N_{l}} P(k) \cdot \text{rel}(k)
$$

where $$N_{rel}$$ is the number of the relevant document returned by the ranker, $$P(k)$$ is the precision calculated at the cut-off positions from 1 to k (i.e. relavant documents among the first $$k$$ documents), $$\text{rel}(k)$$ is an indicator function that is equal to 1 when document at position $$k$$ is a relevant document, and 0 otherwise.


## Vector Space Model with MapReduce

One of possible ways to implement a naive search engine is shown on the diagram below. 
![](https://www.dropbox.com/s/wo2b6amni66djzj/Search%20Engine%20with%20MapReduce.png?dl=1)

The process of creating an index includes

1. Word Enumerator scans the corpus and creates a set of unique words. After that, it assigns a unique id to each word. This task can be implemented with MapReduce. (One can use a hash function to assign IDs and skip this step altogether)
2. Document Count counts the IDF for each term, or simply the number of documents where each particular term appeared. This task can be implemented with MapReduce.
3. Vocabulary is an abstract concept and does not actually require to compute anything. It simply aggregates the result of Word Enumerator and Document Count into one data structure. 
4. The Indexer has to compute a machine-readable representation of the whole document corpus. For each document, the Indexer creates a TF/IDF representation and stores a tuple of `(doc id, vector representation)`. Since each document is processed independently, this can be implemented with MapReduce.

After the index is created, it can be reused multiple times. The Ranking Engine has to create a vectorized representation for the query and perform the relevance analysis

1. The Query Vectorizer is a function that creates the vectorized representation of a query. It can be implemented as a part of the Relevance Analizator. 
2. Relevance Analizator computes the relevance function between the query and each document. This is a brute-force approach. This task can be implemented with MapReduce (the performance depends on available hardware).
3. The index stores the document id and the vectorized representation. The Ranker provides the list of ids, sorted by relevance. The Content Extractor should extract the relevant titles from the text corpus based on the provided relevant ids.

The Query Response contains the list of relevant titles.

## Testing

Depending on the size of the dataset, the processing can take a significant amount of time. To accelerate the development process, it is recommended to perform local testing on a small subset. You can download a small subset from the HDFS cluster using `hdfs dfs -get`. Alternatively, use web interface for HDFS.


## Libraries

Scala is mostly compatible with Java libraries. You can use libraries for JSON deserialization, hashing, data structures. The algorithmic part of the search engine should be original. Your code will be checked for plagiarism, and yes, our plagiarism detection system can easily detect renamed variables and permutations of lines.


## Broadcast objects

Spark is a framework for distributed computations. Sometimes you need to distribute some data between all executors. Use [broadcast objects](https://spark.apache.org/docs/latest/rdd-programming-guide.html#broadcast-variables) to achieve this.

