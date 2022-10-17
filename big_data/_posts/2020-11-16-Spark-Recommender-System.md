---
layout: post
title: Spark Recommender System
categories: [big-data]
tags: [Recommender]
mathjax: true
show_meta: true
published: true
---

The goal of this task is to finalize the implementation of a movie recommendation system. In the process, you will get more experience with programming in Scala and working with RDDs.

**Dataset**: Modified [MovieLens dataset](https://edisk.university.innopolis.ru/edisk/Public/Fall%202019/Big%20Data/Recommender%20System/) ([Mirror](https://www.dropbox.com/sh/y5uck2kbzizaes6/AACpapXw-JMmLJB_xCGOZOFCa?dl=0))

**Project Template**: Download the [project](https://edisk.university.innopolis.ru/edisk/Public/Fall%202019/Big%20Data/Recommender%20System/) and try compiling it. Compilation should complete without errors. ([Mirror](https://www.dropbox.com/sh/y5uck2kbzizaes6/AACpapXw-JMmLJB_xCGOZOFCa?dl=0))

**Extra files**: Download [list of movies](https://www.dropbox.com/s/79d63pnsnuy6zr3/for-grading.tsv?dl=1) for collecting user preferences 

## Inspecting and Running The Template

Inspect the project template content. 

```
spark-recommendation
├── build.sbt
├── for-grading.tsv
└── src
    ├── main
    │   └── scala
    │       ├── Grader.scala
    │       └── MovieLensALS.scala
    └── test
        └── scala
```
- `build.sbt` contains build configuration and dependencies.  Make sure that the correct [spark and scala](https://spark.apache.org/docs/latest/#downloading) (read the documentation or run `spark-submit --version`) versions are specified here
- `for-grading.tsv` this file is used for collecting user ratings 
- `Grader.scala` for reading user preferences
- `MovieLensALS.scala` contains main class for training recommendation system

Try to compile this project with sbt or [create an artifact with your IDE](https://www.jetbrains.com/help/idea/creating-and-running-your-first-java-application.html#create_class). When using IDE, make sure all dependencies are in MANIFEST. More information about this in the Appendix.

Try to run application with 
```
spark-submit --master yarn spark-recommendation.jar hdfs:///movielens-mod -user false
```

## Data Description

The dataset includes two files: `movies2.csv` and `ratings2.csv`. The first contains the list of movie titles in the format
```
movieId,title,genres
```
for example
```
72,Kicking and Screaming (1995),Comedy|Drama
73,"Misérables, Les (1995)",Drama|War
```

The ratings data are stored in the second file and have the format 
```
userId,movieId,rating,timestamp
```
for example
```
1,2253,3.5,1112486122
1,2288,4.0,1094786077
```

The data is also available in HDFS at `/movielens-mod`.

## General description

Your goal is to implement a movie recommendation system. The system takes data in the format `Rating(userId, movieId, rating)`, and tries to learn a model of a user, based on graded movies. You need to format the data as needed using RDD transformations. For more information about class `Rating` read [here](https://spark.apache.org/docs/2.3.2/api/java/org/apache/spark/mllib/recommendation/Rating.html).

In Spark, recommendation system can be built with the [`ALS` (Alternating Least Squares)](https://spark.apache.org/docs/2.3.2/api/java/org/apache/spark/mllib/recommendation/ALS.html) class that has simple [`train()`](https://spark.apache.org/docs/2.3.2/api/java/org/apache/spark/mllib/recommendation/ALS.html#train-org.apache.spark.rdd.RDD-int-int-) and [`predict()`](https://spark.apache.org/docs/2.3.2/api/java/org/apache/spark/mllib/recommendation/MatrixFactorizationModel.html#predict-org.apache.spark.rdd.RDD-) interfaces.

The predictor takes the data in format `(userId, movieId)` and returns `Rating(userId, movieId, rating)`. To make the results human-readable, we need an additional structure that stores the mapping of `movieId` to `movieTitle`

To evaluate the benefit of this system, we compare the result with a baseline. The baseline for this task is the average rating for a given movie. We need a data structure that stores the mapping of `movieId` to `averageRating`.

Read the code. Understand how the program works.

## Complete the code

Implement `parseTitle` and `rmse(test: RDD[Rating], prediction: scala.collection.Map[Int, Double])`

```
spark-submit --master yarn spark-recommendation.jar hdfs:///movielens-mod -user false
```


## Run with Your Movie Preferences
```
spark/bin/spark-submit --master yarn spark-recommendation.jar hdfs:///movielens-mod -user true
```
executing this command will allow you to specify your own movie preferences and get recommendations after the model has finished training.

:::info
The dataset is not very large. You can run the application locally if you have sufficient resources.
:::

## Post-processing of Recommendations

Currently, the list of recommendations can include the movies that the user has already graded. Modify the program such that these movies are filtered. You can use either set difference or RDD's filter methods.

## Load Your Movie Preferences

Your movie preferences were saved into the file `user_rating.tsv`. Write a method or class that will load this data instead of surveying the user every time.

## Change the Rank of the Model

Try different model ranks. A higher rank leads to a more complex model. Compare the difference between baseline and prediction for a model with a higher rank. Evaluate the quality of the proposed movies subjectively.

## Extra Filtering

You might have noticed that the recommendations are not very great. The reason is that the model cannot compute confident representations for infrequent movies. Modify the program in such a way that low-frequency items (movies with less than 50-100 ratings) are excluded from the training and recommendation process. 

## Reference
- [Help with Scala](https://learnxinyminutes.com/docs/scala/)
- [Scala Book](https://booksites.artima.com/programming_in_scala_3ed)
- RDD partitioning [1](https://medium.com/parrot-prediction/partitioning-in-apache-spark-8134ad840b0) 
- Stack overflow during execution [1](https://issues.apache.org/jira/browse/SPARK-1006)
- History Server [1](https://spark.apache.org/docs/latest/monitoring.html)


## Self-Check Questions

- What does `sc.parallelize` do?
- What does `collectAsMap` do?
- What is the difference between `foreach` and `map`?
- What is pattern matching in Scala?


## Troubleshooting

### Sbt compilation error
There are issues compiling scala libraries with Java 9 and above. If you see an error during compilation, try downgrading to Java 8.

### Missing Winutils
There is a known issue with Spark on Windows, that it requires winutils.exe. The solution can be found [here](https://stackoverflow.com/questions/35652665/java-io-ioexception-could-not-locate-executable-null-bin-winutils-exe-in-the-ha).

### Memory Issues
If you experience issues with memory, try to partition dataset into larger chunks
```
val data = data.repartition(number_of_partitions)
```
Creating more partitions is equivalent to reducing the size of input data per partition, and reducing memory requirements for map tasks.

## Appendix

### Including Dependencies

When you configure artifact with IDE, you need to specify the dependencies. There is absolutely no need to include all the dependencies in the jar container, and you should merely add dependencies to the manifest.