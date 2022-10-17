---
layout: post
title: Spark MLLib. Sentiment Classification
categories: [big-data]
tags: []
mathjax: true
show_meta: true
published: true
---

[Spark MLlib](http://spark.apache.org/docs/latest/ml-guide.html) has a variety of different ML algorithms implemented. Due to limited resources, we will restrain ourselves from using sophisticated methods and use mainly the simplest algorithms. The focus of this lab is [pipelines](http://spark.apache.org/docs/latest/ml-pipeline.html), [transformations](http://spark.apache.org/docs/latest/ml-features.html), [estimators](http://spark.apache.org/docs/latest/ml-classification-regression.html), and [model selection](http://spark.apache.org/docs/latest/ml-tuning.html).

## Dataset Description
For the task of sentiment classification we are going to use [Twitter dataset from Kaggle](https://edisk.university.innopolis.ru/edisk/Public/Fall%202019/Big%20Data/Sentiment%20Data/) ([Original](https://www.kaggle.com/c/twitter-sentiment-analysis2/data)). This dataset contains 100k examples. The data is stored in CSV format. Data is very irregular and requires preprocessing
```
ItemID,Sentiment,SentimentText
1,0,                     is so sad for my APL friend.............
2,0,                   I missed the New Moon trailer...
3,1,              omg its already 7:30 :O
```

## Spark ML Pipelines

Spark ML allows creating custom [data processing pipelines](http://spark.apache.org/docs/latest/ml-pipeline.html). These pipelines are used to create data flow DAG. Do not confuse this dataflow DAG with other DAG's that you can find in popular machine learning libraries like TensorFlow. In Spark, each stage of the pipeline can be either a custom processing function (classical programming) or an ML stage. ML stages are trained independently from each other. Further, we are going to discuss feature extraction tools available in Spark ML and later we will see how to fit all of them into a pipeline. 

We are going to build a data processing pipeline with DataFrames. In this case, each pipeline stage accepts a DataFrame column as input and requires to specify the output column. For example

```scala
val stageInstance = new PipelineStage()
      .setInputCol("InColumn")
      .setOutputCol("OutColumn")
      // this is an generic example 
```

## Reading the Data

Twitter data is provided in CSV format. Spark comes with a set of tools for data import.
```scala
val spark = SparkSession
      .builder()
      .appName("Some Name")
      .getOrCreate()
```

After you have instantiated the spark session object, you can read CSV with a standard method
```scala
val data = spark.read.format("csv")
      .option("header", "true")
      .load(path)
```

This method will return a [DataFrame](https://spark.apache.org/docs/latest/sql-data-sources-load-save-functions.html). Working with DataFrames in Spark is similar to DataFrames in Python - the data is organized in a tabular form. You can slice a column using array indexing syntax `data("ColumnName")`.


## Feature Extraction 

Sentiment Classification, in most cases, is a task of binary classification. Usually, the goal is to discern negative comments from neutral or positive (or other combinations of these three). 

Further, several techniques for feature extraction are described for the case of Spark ML DataFrames API. 

<p>
<img src="https://i.imgur.com/yBkrWse.png" alt="drawing" width="550"/>
</p>

### Preprocessing

The preprocessing stage involves text normalization. People can write in very different styles, and when it comes to understanding the meaning of a text snippet, it is easier to create a model when this variation is removed. Consider a simple example from [Twitter Sentiment dataset](https://www.kaggle.com/c/twitter-sentiment-analysis2/data). 
```
Juuuuuuuuuuuuuuuuussssst Chillin!!
```
The meaning and sentiment are easy to understand for a human, but a computer can struggle with this example. What we ideally need to do is to extract the semantic meaning from this text, along with the style. One of the possible normalizations is to lowercase and remove all repetitive characters 
```
just chiling !
```
When you work with Spark's DataFrame, this is done by applying `map` transformation to one of the columns. This can be done with the help of [User Defined Functions](https://spark.apache.org/docs/latest/sql-ref-functions-udf-scalar.html). Their primary benefit is the compatibility with Spark SQL syntax. A UDF that removes repetitive characters and converts characters to lower case can look like this

```scala
import org.apache.spark.sql.functions.udf

val removeRepetitive = udf{ str: String => str.replaceAll("((.))\\1+","$1").trim.toLowerCase()}

val noRepetitiveData = data.withColumn("Collapsed", removeRepetitive('SentimentText))
```

The notation `'SentimentText` signifies a specific column with a given name.

Executing the code above will return a new DataFrame with a new column appended.

Alternatively, you can [implement your own pipeline stage](https://www.oreilly.com/learning/extend-spark-ml-for-your-own-modeltransformer-types). 

### Tokenization

Spark ML provides two pipeline stages for tokenization: [Tokenizer and RegexTokenizer](https://spark.apache.org/docs/latest/ml-features.html#tokenizer). The first one converts the input string to lowercase and then splits it on whitespace characters and the second extracts tokens using the provided regex pattern.

```scala
import org.apache.spark.ml.feature.{Tokenizer, RegexTokenizer}


val tokenizer = new RegexTokenizer()
      .setInputCol("PreprocessedSentences")
      .setOutputCol("Tokens")
      .setPattern("\\s+")
val wordsData = tokenizer.transform(sentenceData)

```

### Stop Words Removal
Spark ML also provides a [pipeline stage for stop words removal](https://spark.apache.org/docs/latest/ml-features.html#stopwordsremover).

### TF/IDF Features

Spark ML provides several stages for document vectorization. Two of them are [HashingTF](https://spark.apache.org/docs/latest/ml-features.html#tf-idf) and [CountVectorizer](https://spark.apache.org/docs/latest/ml-features.html#countvectorizer). Both return a numerical vector that can be treated as input features for a classifier. The difference is that `CountVectorizer` enumerates the unique words first, and, therefore, requires calling `fit`.

```scala
import org.apache.spark.ml.feature.{HashingTF, IDF}

val hashingTF = new HashingTF().setInputCol("words").setOutputCol("tf").setNumFeatures(2000)
```
This will create a vector representation of texts. This vector stores the term frequencies in the sparse data format `(vector_size, positions, values)`.

```scala
val featurizedData = hashingTF.transform(wordsData)
// given input data was 
// >> data.select("tokens").collect().foreach(println)
// [Array(a, great, day)]
// [Array(a, great, day, day, a)]
//
// The output will be 
// >> data.select("features").collect().foreach(println)
// [(2000,[165,467,1768],[1.0,1.0,1.0])]
// [(2000,[165,467,1768],[1.0,2.0,2.0])]
```

Be aware, if you use `CountVectorizer` the size of your vocabulary will grow with the amount of training data. Each new term adds additional parameters to your model. Often a reasonable approach to constrain dimensionality is to use N most frequent tokens. Refer to tokenizer documentation for details. 

IDF can be computed as 
```scala
val idfModel = new IDF().setInputCol("tfFeatures").setOutputCol("tfidfFeatures").fit()
val rescaledData = idfModel.transform(featurizedData)
// The output will be 
// >> data.select("tfidfFeatures").collect().foreach(println)
// [(2000,[165,467,1768],[0.0,0.0,0.0])]
// [(2000,[165,467,1768],[0.0,0.0,0.0])]
// Since logarithm is used for computing IDF, if a term appears in all documents, its IDF value becomes 0. That is why the current output is 0.
```

The formula that is used for computing IDF is given in the [documentation](https://spark.apache.org/docs/latest/ml-features.html#tf-idf). The output of the IDF stage is TF/IDF vector. The resulting features are suitable for use in every classifier built into Spark MLlib. 


### Optional: Including NGrams

<p>
<img src="https://i.imgur.com/NLgewhb.png" alt="drawing" width="550"/>
</p>

Sometimes, including N-Grams can improve model accuracy. One of the possible architectures is shown in the figure above. Some auxiliary stages can be added, such as stop words removal (although preserving stop words can be beneficial for n-grams, should be decided with cross-validation).

1. N-Gram Extraction and Vectorization
Once again, Spark ML has [neccessary methods in-place](https://spark.apache.org/docs/latest/ml-features.html#n-gram). Keep in mind, that the number of possible N-Grams grows very fast. Given the limitations on the available memory and the complexity of the model, you should limit the size of N-Gram vocabulary. Otherwise, your model can easily overfit.
2. Feature Assembler
Spark ML's pipeline stage [VectorAssembler](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler) helps to concatenate several vectors together. Thus, you can concatenate vectorized tokens and vectorized N-Grams, creating a single feature vector that you are going to pass to the classifier.


### Optional: Word Embeddings

<p>
<img src="https://i.imgur.com/0CBSj46.png" alt="drawing" width="550"/>
</p>

An alternative to using token-based vectorization, one can resort to word embeddings. Spark ML provides a [pipeline stage](https://spark.apache.org/docs/latest/ml-features.html#word2vec) for converting an array of tokens into a vector using embeddings. Leaving the details of how exactly embeddings work, consider how the pipeline stage works:

1. Given the training data (arrays of tokens), the stage trains word embeddings for each token that occurs more than 5 times (default value) in the dataset. The quality of the vectors depends partially on the number of iterations specified for the stage (usually, the more, the better)
2. After the vectors are trained, each entry (array of tokens) is combined. Spark ML uses simple summation over word vectors (check the method [`transform(dataset: Dataset[_])`](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/feature/Word2Vec.scala)).
3. Now, each entry is represented by a vector of predefined dimensionality (dimensionality is a parameter of the Word2Vec stage. Refer to the documentation for details), and these vectors are passed directly to the classifier.

Since this feature extraction model involves training word vectors, it can take a significantly longer time to train. 

## Classification Method

### Logistic Regression

Spark ML provides many classification methods. Since we discussed how to extract numerical features, it is recommended to resort to methods that can be optimized by gradient descent. The most simple model is [LogisticRegression](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression). The documentation has a thorough example of the usage. It can be also applied for the multi-class case. 

```scala
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, LogisticRegression}

val lr = new LogisticRegression()
      .setFamily("multinomial")
      .setFeaturesCol("tfidfFeatures")
      .setLabelCol("Sentiment")
val lrModel = lr.fit(training)
```

Always evaluate your model after training. Feed some data to the model and look at the output. For LR, the output is the `probabilities` and `prediction` columns of the DataFrame. If the `probabilities` are the same or almost the same for different examples --- your model did not learn. The stopping criteria for training the model is reaching the maximum number of iterations or the convergence threshold (the method `setTol`). 

### Optional: Multilayer Perceptron

It is also possible to create a stage with a [Neural Network](https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier). You can specify the number of layers, layer sizes, learning rate, and convergence tolerance. 

## Training

### Assembling Pipeline

So far we have tried to apply different transformations interactively. This included calling methods `fit()` and `transform()`. Note that we do not need to do this for every stage when we are working with pipelines. Simply, create new pipeline stages (using the operator `new`) and assemble the pipeline. Later, call `fit()` and `transform()` for the entire pipeline. 

Assembling a pipeline in Spark is as easy as creating a single object instance

```scala
import org.apache.spark.ml.Pipeline

// pipeline example
val pipe = new Pipeline()
.setStages(Array(
  tokenizer, // split original sentence on whitespaces
  // this first stage will capture punctuation and emoji as well as words
  wordTokenizer, // tokenize original sentence in word tokens
  // this second tokenized will be used to create ngrams
  // we want to reduce the number of ngrams, and therefore filter all non word charactes
  ngram, // create ngrams from word tokens
  tokenVectorizer, // map token into features
  ngramVectorizer, // map ngrams into features
  assembler, // concatenate feature vectos
  classifier // add classification stage 
))
```

Then you can train all trainable stages with a single call
```scala
val model = pipe.fit(twitterData)
```
where `TwitterData` is the input to your pipeline.

### Distributed Hyperparameter Search

In many situations, the data in your pipeline can be processed with the help of only a single computer. `transform` method runs locally in every worker, and the model sizes should fit one single worker's memory. This way, Spark helps us to process the large quantities of data in a distributed way, but it does not allow us to create distributed machine learning models.

Before you saw that you can specify some of the parameters for the stage during instantiation. Alternatively, you can create a parameter map and pass it to the pipeline's `fit` method.

```scala
import org.apache.spark.ml.param.ParamMap

val paramMap = new ParamMap()
      .put(tokenVectorizer.vocabSize, 10000)
      .put(ngramVectorizer.vocabSize, 10000)
      .put(classifier.tol, 1e-20)
      .put(classifier.maxIter, 100)
      
val model = pipe.fit(twitterData, paramMap)
```
this way you can keep all parameters in a single object, and set them from a configuration file in a convenient way.

Another benefit of Spark is that it allows utilizing the cluster resources to perform an extensive hyperparameter search. Spark allows the training of several models simultaneously. For this, you need to create a [parameter grid map](https://spark.apache.org/docs/latest/ml-tuning.html#model-selection-aka-hyperparameter-tuning)
```scala
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

val paramGrid = new ParamGridBuilder()
      .addGrid(tokenVectorizer.vocabSize, Array(10000, 20000))
      .addGrid(gramVectorizer.vocabSize, Array(10000, 15000))
      .addGrid(lr.tol, Array(1e-20, 1e-10, 1e-5))
      .addGrid(lr.maxIter, Array(100, 200, 300))
      .build()
```

### Cross Validation

Similarly, you can parallelize model training in a [k-fold cross validation prosedure](https://spark.apache.org/docs/latest/ml-tuning.html#cross-validation).

```scala
val cv = new CrossValidator()
      .setEstimator(pipe)
      .setEvaluator(new BinaryClassificationEvaluator()
      .setRawPredictionCol("prediction")
      .setLabelCol("Sentiment"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)  // Use 3+ in practice
      .setParallelism(2)
```

This class will train several models as specified by the parameter grid map and select the best set of parameters based on the selected Evaluator. After you instantiated CrossValidator, you use it to fit the model
```
val model = cv.fit(twitterData)
```

## Applying Model

You can use trained model to process new data
```scala
val result = model.transform(twitterTestData)
```

## Configuring to Run on Cluster

After you are done developing the project and testing it on a small data subset, it is time to launch a full-scale training. Compile your applications, copy your jar to the cluster (use one of your team accounts from the previous assignment). Run `spark-submit`. Set additional parameters if necessary
```
spark-submit  \
    --master yarn \
    --deploy-mode client \
    --driver-memory 500m \
    --executor-memory 500m \
    --executor-cores 1 \
    my-app.jar arg1 arg2
```

You can deploy your JAR to the cluster with the following command

```bash
spark-submit --master yarn --deploy-mode client path/to/jar hdfs://twitter/
```
You can check the status at `http://namenode:8088`

## Self-Check Questions
- Can Spark MLlib process images?
- What are the limitations of Spark MLlib