---
layout: post
title: Stream Processing With Spark
categories: [big-data]
tags: []
mathjax: true
show_meta: true
published: true
---

Spark Stream creates an abstraction over streams that process stream data almost identically to RDDs or DataFrames. The most common (easy to work with) stream format is Discretized Stream (`DStream`). Alternatively, you can convert your stream to Spark `DataFrame` and process it using SQL Operations. All the details of how to create a stream object, the list of transformations implemented for streams, ways to convert `DStream` to `DataFrame` are provided in the official [programming guide](https://spark.apache.org/docs/latest/streaming-programming-guide.html). Read it through before beginning your work.

One of the common stream formats is `DStream` containing lines. This format is almost identical to `RDD[String]`. The list of available transformations for `DStream` can be found [here](https://spark.apache.org/docs/latest/streaming-programming-guide.html#transformations-on-dstreams). Each entry in the stream is a line - e.g.

```
thrilled about being at work this morning
```

This line is a single tweet.

After receiving this line, you can apply a series of transformations. For our task, we are mostly interested in classifying tweets by sentiment.

### Preprocessing Steps

Familiarize yourself with the format of the stream. We recommend that you turn to [spark documentation](https://spark.apache.org/docs/latest/streaming-programming-guide.html#transformations-on-dstreams). The preprocessing is as simple as writing several map functions. The goal of preprocessing is usually data normalization.

### Building a Sentiment Classifier

The goal of the sentiment classification task is to figure out positive and negative (or neutral) tweets.

The most widely used application is product review analysis. Sentiment analysis is one of the most common tools for assessing the perception of a company by its customers. In most of the cases, the problem of sentiment analysis can be reduced to classification into two or three classes. For more information about sentiment classification, read [this note](https://hackmd.io/s/BJdE2mZA7).

#### Selecting Datasets

To train a sentiment classifier, one needs to collect a labeled dataset. There are several free datasets that you can use.

1. [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
This is a dataset for binary sentiment classification (positive or negative) containing a set of 25,000 highly polar movie reviews for training, and 25,000 for testing
2. [Sentiment Tree Bank](https://nlp.stanford.edu/sentiment/code.html)
Contains sentences with continuous sentiment measure: 5 classes can be distinguished by mapping the positivity probability using the following cut-offs:
```[0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]```
for very negative, negative, neutral, positive, very positive, respectively.
3. [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)
Contains binary classified reviews from imdb.com, amazon.com, and yelp.com.
4. [Twitter Sentiment](https://www.kaggle.com/c/twitter-sentiment-analysis2/data)
Contains messages from twitter labeled with one of three classes: neutral, positive, negative.

You can use one of these datasets to train your model or even merge these datasets to improve the results. But remember that the stream comes from twitter.

#### Feature Extraction

To get good results in the classification task you need to provide the algorithm with a suitable set of features. Read the [documentation](https://spark.apache.org/docs/latest/ml-features.html) about methods available in Spark for feature extraction and feature transformation. Some useful methods are [TF/IDF](https://spark.apache.org/docs/latest/ml-features.html#tf-idf), [CountVectorizer](https://spark.apache.org/docs/latest/ml-features.html#countvectorizer), [FeatureHasher](https://spark.apache.org/docs/latest/ml-features.html#featurehasher), [Word2Vec](https://spark.apache.org/docs/latest/ml-features.html#word2vec), [Tokenizer](https://spark.apache.org/docs/latest/ml-features.html#tokenizer), [StopWordRemover](https://spark.apache.org/docs/latest/ml-features.html#stopwordsremover), [n-gram](https://spark.apache.org/docs/latest/ml-features.html#n-gram), and other.

#### Selecting Classifier Model

There are several ways to approach the classification problem. In this homework, you can either use standard Spark methods or implement your own classifier. There is a [multitude of algorithms](https://spark.apache.org/docs/latest/ml-classification-regression.html) available in Spark. All you need to do is to select the ones useful for your task. Alternatively, you can implement your own algorithm. You need to try at least two algorithms from standard ones or implement one of them on your own.

<p align="center">
<img src="https://i.imgur.com/UAGMFFb.png" width=500>
</p>


You need to train and compare models using [F1 scores](https://en.wikipedia.org/wiki/_score). Then you need to run the models on the stream and provide to output files with your sentiments classes. 

In the report, you should elaborate more on results - provide outputs, provide scores (for train and stream data), and report which model achieves the best performance.

Remember that you can save a trained model and load it later. 


### Preparing Results for Output

Monitor the Twitter stream throughout a day, collect tweets, classify, and store the results. After that, you need to evaluate your classifier's correctness manually. Look through collected data and manually verify the correctness of the classifier (take a subsample of the twitter stream and estimate the actual classification quality). Provide some estimates for precision and recall for your classifier. 

The output files should contain a log of incoming data and decisions made in the following format (CSV). (One file for each model).

```
Time, Input String 1, class
Time, Input String 2, class
```

You need to store all output files in your groups' folder on HDFS. Describe the data formats used in your log files in your report. 

The implemenation can be found [on Github](https://github.com/VitalyRomanov/twitter-sentiment).