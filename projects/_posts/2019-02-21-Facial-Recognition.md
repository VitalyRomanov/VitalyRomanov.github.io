---
layout: post
title: "Facial Recognition"
date: "2019-02-21"
description: ""
category:
  - Machine Learning
tags:
  - Facial Recognition
mathjax: true
published: true
---

The classical way to approach the problem of face recognition is to collect the data with different faces of people and build a classifier that will tell the difference between people. The problem with this approach is that such models are not very portable. Adding new people requires retraining the model and the behavior on people not present in the dataset is not well defined. Another approach to this problem is *Unconstrained Classification* where the number of target classes is not specified at the training time.

## Unconstrained classification

The classical classification tasks usually formulated in a way where one has a labeled dataset and the set of target labels is fully represented in the data. Moreover, the quality of classification often depends on the number of training examples. Contrary to that, the problem of unconstrained classification does not assume the finite number of classes. 

For many methods, the output of a classifier is a vector with confidence scores about the current sample's class. More formally, the decision is made with the following rule

$$
\hat{c} = \underset{c\in C}{\operatorname{argmax}}R(x, c) 
$$

where $C$ is a set of all classes, $x$ is the current data sample and $R$ is a ranking function (the greater $R$ the more our confidence that $x$ belongs to $c$).

Unconstrained classification problems should have a different way of making a decision since the set $C$ is not completely defined.

The possible solution to this problem is to define a continuous space and dedicate different region of this space to specific classes. This way, the new classes can be added on the fly, since all we need is to decide what a particular region of space represents. As you can guess this approach is related to the problems of clustering and nearest neighbor search

But simple nearest neighbor classification will not work because it will assign a class even when the new sample is really far from any of the centroids of any of the known clusters. The simplest solution is to add some threshold. The classification rule now is
$$
\begin{equation}
  \hat{c}=\begin{cases}
    \underset{c\in C}{\operatorname{argmax}}R(x, c), & \text{if } \exists c \in C: R(x, c) > T,\\
    \emptyset, & \text{otherwise}.
  \end{cases}
\end{equation}
$$

where T is the similarity threshold. The meaning of this classification rule is similar to the first one, except we leave ourselves the opportunity to refuse to assign any class, in case our confidence level is not sufficient. We make this decision by requiring the correct class to have similarity of at least $T$. Algorithmically, we simply need to find argmax, and then check that similarity for this class satisfies the threshold.

## Triplet Loss (Margin Loss)

> An embedding is a representation of a topological object, manifold, graph, field, etc. in a certain space in such a way that its connectivity or algebraic properties are preserved. For example, a field embedding preserves the algebraic structure of plus and times, an embedding of a topological space preserves open sets, and a graph embedding preserves connectivity.
> *Source: [Wolfram](http://mathworld.wolfram.com/Embedding.html)*

For the task of face classification, we want to find an embedding of a face that captures special facial properties so that it is easy to tell different faces apart by looking at some similarity measure. In other words, we want an embedding function $f$ that projects an image $x$ into the embeddings space, where different faces are at least as far from each other as the threshold value $T$. Since it is hard to come up with such a function $f$, we try to learn it. The only thing we care about is the interpretation of the distance between embeddings, and this will be the only criteria that we enforce using the optimization loss. Given an image, we call it an anchor, we want to minimize the distance between other images of the same face, call it positive examples, and we want to maximize the difference with other faces. Formally

$$
||f(anc) - f(pos)||^2 + \alpha \leq ||f(anc) - f(neg)||^2
$$

where $anc$ is the anchor image, $pos$ and $neg$ are positive and negative images correspondingly. If we remember the notion of similarity function $R$ from the previous section, for this loss it is defined as follows

$$
R(x, c) = - ||f(c) - f(x)||^2
$$
where maximum similarity is 0, and the class itsef is represented by the anchor image.

This property is captured in the equation for *Triplet Loss*

$$
loss = \frac{1}{N} \sum_{i=1}^N \left[||f(anc_i) - f(pos_i)||^2 + \alpha - ||f(anc_i) - f(neg_i)||^2 \right]_{+}
$$

Here, we make sure that there is a penalty as long as there are negative samples that are closer to the anchor than the margin value $\alpha$. The operator $[x]_+$ is equivalent to `max(0, x)`.

## Architecture

For this task, we will try to benefit of image recognition by loading model pretrained on ImageNet. ImageNet is an image classification task with 1000 distinct classes. Even though it tries to solve a different problem, we can benefit from the fact that the model learned how to interpret an image. Then we simply discard last layers that do the classification and add our new layers that will encode facial features.

![](https://software.intel.com/sites/default/files/managed/58/24/transfer-learning-fig2-schematic-depiction-of-transfer-learning.png)
*Transfer learning. Source: [Intel](https://software.intel.com/en-us/articles/use-transfer-learning-for-efficient-deep-learning-training-on-intel-xeon-processors)*

### CNN: Loading Inception V3 ([Download](https://www.dropbox.com/s/b65bezg84a8mt1i/InceptionV3.pb?dl=1))

For our problem, we will import the weights of a CNN with architecture Inception V3 pretrained on ImageNet ([GitHub page of the CNN project](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)).

For your convenience, we simplified the process of loading the model, discarded from the last layers of the network, and exported the model to a [frozen graph definition](https://www.tensorflow.org/guide/extend/model_files#graphdef). The file with the model `InceptionV3.pb` can be found [here](https://www.dropbox.com/s/7qdm7ptr53nknr8/dataset.zip?dl=1). The input to the graph is a tensor named `Model_Input` with dimensions `[None, 299, 299, 3]`, and the output named `Model_Output` with the shape `[None, 2048]`. The weight can be loaded with the following piece of code

```python
pretrained_graph_path = 'InceptionV3.pb'
with tf.Session() as sess:
  with tf.gfile.GFile(pretrained_graph_path,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
```

After that, the placeholder object and the CNN embedding tensor can be obtained
```python
input_ = tf.get_default_graph().get_tensor_by_name("Model_Input:0")
cnn_embedding_ = tf.get_default_graph().get_tensor_by_name("Model_Output:0")
```

### Adding new layers

In order to adapt the model to facial recognition, add new layers to the model. The baseline is the following:
- Dence layer with 512 units with sigmoid activation
- Dence layer with 256 units with sigmoid activation, the output is normalized to 1 (use `tf.l2_normalize`)
- Dence layer with 128 units with tanh activation, the output is normalized to 1

Remember that for each minibatch you have a batch of anchor images and batches corresponding to positive and negative images. We recommend creating three placeholders for these batches to simplify calculations. For the training process, there is really no need to connect the last layers with the CNN model, since we do not need to train CNN. Bypassing CNN with placeholders will allow us to accelerate computations. All three batches with anchors, positive and negative images should pass through the same network. This approach is called *Siamese Architecture*

![](https://www.oreilly.com/library/view/tensorflow-1x-deep/9781788293594/assets/15b0d10f-3abe-4254-87dd-e3cb5ad93494.png =400x)
*Source: [Orelly](https://www.oreilly.com/library/view/tensorflow-1x-deep/9781788293594/b109c39d-4c68-45e1-90de-c9c307498783.xhtml)*

As you can see in the picture above, the data $X_1$ and $X_2$ follow through different networks, but weights are shared. 

```
input_1 = tf.placeholder(tf.float32, [None, 100])
input_2 = tf.placeholder(tf.float32, [None, 100])

with tf.variable_scope("siamese") as scope:
    out_1 = self.network(input_1)
    scope.reuse_variables()
    out_2 = self.network(input_2)
```

After you assemble the model, define loss and create Adam optimizer, you should see a graph in Tensorboard similar to the one below. 

![](https://i.imgur.com/4O5U0Uh.png)



The subgraph `face_feature_extraction` contains the siamese architecture (in case you implemented the baseline architecture).


![](https://i.imgur.com/UlshQE8.png)


To visualize your graph, make use of the following code snippet
```python
with tf.Session() as sess:
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("./tf_summary", graph=sess.graph)
```
and then launch tensorboard from the working directory of your project
```bash
tensorboard --logdir ./tf_summary/
```

### Inference and Selecting Threshold

So far there we stated the existence of two thresholds: $T$ and $\alpha$. The first one is used to make the true class and the second (margin) to improve the construction of the face embedding space. These two values should be distinct because the margin $\alpha$ is merely a suggestion baked into the loss function, and there is no way to guarantee its strictness. It does not ensure that positive samples will fall into the hyper-sphere of radius $\alpha$ and all negative samples will be outside of it. Due to this, we need to come up with another threshold value to make a decision about the class of the current image, and this threshold will be $T$.

For inference, you need to compare two images and decide whether they are the same person or not. The decision should be made based on some threshold value. We selected the threshold of -0.8 based on the model performance after 500 epochs (assuming $R$ is defined the same way as in the section *Triplet Loss*). The actual threshold should be selected after the model was trained for hundreds of hours, the threshold is chosen with k-fold cross-validation or with a held-out dataset. 

## Data Description

For training the model we are going to use preprocessed [LFW](http://vis-www.cs.umass.edu/lfw/) dataset. All the faces were aligned and cropped. You can read more about this [here](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8).

### Dataset Structure

The dataset structure is preserved and has the following form
```
# Directory Structure
# ├── Tyra_Banks
# │ ├── Tyra_Banks_0001.jpg
# │ └── Tyra_Banks_0002.jpg
# ├── Tyron_Garner
# │ ├── Tyron_Garner_0001.jpg
# │ └── Tyron_Garner_0002.jpg
```
Where for every person there is a folder with the photos of faces. There are people with only one photo available: do not include them in the training process.

### Loading Images

Cropped images in our dataset have lower resolution than required by the CNN network. For this reason, the image should be upscaled after importing.

```python
import cv2

im_size = 299 
load_image = lambda im_path: cv2.resize(cv2.imread(im_path), (im_size, im_size))

path = "img.jpg"
img = load_image(path)
```

For this example, `img` is a `numpy` array and can be passed directly to `tensorflow` with `feed_dict`. The shape of the image in this example would be `(299,299,3)`. To satisfy the placeholder dimensionality make use of `np.newaxis`. To stack several images in minibatch use `np.stack`.

### Test Data ([Download](https://www.dropbox.com/s/loz4ijaxcor3l5l/test_set.csv?dl=1))
Test set is provided in the form of csv file

| Anchor | Positive | Negative |
| -------- | -------- | -------- |
| Vincent_Brooks/Vincent_Brooks_0002.jpg     | Vincent_Brooks/Vincent_Brooks_0006.jpg     | Gerhard_Schroeder/Gerhard_Schroeder_0007.jpg     |

We are going to evaluate the training based on classification accuracy and test set loss. The test set table contains 400 rows. This implies 800 pairs to compare and 800 decisions (same/different) to make. *Test set accuracy* is the accuracy score on those 800 comparisons.

### Lazy Loading and Caching

Images occupy large a amount of memory. For this reasons it is advised to keep them from loading until they are needed and discard them after the processing is done. Moreover, we do not really need to work with the original images at all, since our new layers require only CNN representation. You can keep a cache of images in memory (or on disk), where you store CNN representation for every image.

The simplest cache in python could be implemented with a dictionary. 

```python
def cache_images(batch_pahts, cache):
    for image_path in batch_paths:
        if image_path not in cache:
            img = load_image(image_path)
            cache[image_path] = CNN(img)
            
            
def get_from_cache(batch_paths, cache):
    return np.stack([cache[image_path] for image_path in batch_paths], axis=0)


cache = dict()
minibatch_paths = get_minibatch_paths()
cache_images(minibatch_paths, cache)
minibatch = get_from_cache(minibatch_paths, cache)
pickle.dump(cache, open("img_cach_file.pkl", "wb"))
```

### Generating Minibatches

Training a good embedding function requires us to iterate through all possible combinations of anchors, positive and negtive images. Clearly, this is very computationally expensive. Instead, we can approach this problem with statistics, and assume that a sample large enough from a population will be sufficient to represent some properties of this population with an acceptable amount of error. 

The simplest policy for creating minibatches is the following:
1. During one epoch we will iterate over all anchors, so achors can be sampled sequentially or randomly
2. For every anchor we need to select one positive example. Given that some people have more than two images available, we randomly sample the positive image. Make sure anchor and positive example do not match.
3. For every anchor we need to sample a negative example. This could be any image of a different person.

## Advanced Topics

### Adaptive Minibatch Sampling

To help the model to train faster you can give it triplets that are harder to solve. For example, among all positive samples, select the one that is the most different, among all negative samples, select the one that is the most similar.

You can implement it by processing all the images after each epoch and storing a result in a structure that supports fast knn queries. 

Then, you will be able to find closest and farthest positive and negative examples efficiently.

Using the closest or the farthest examples is not the most efficient strategy. It is better to create a weighted distribution and sample closest negative examples or farthest positive examples more frequently. For more read about python's `random.choice`.

You can read more about positive and negative sampling in the [original paper](https://arxiv.org/pdf/1503.03832.pdf).

### Annealing Learning Rate

You can help you model to learn better by gradually reducing the learning rate. One of the ways to implement this is to create a placeholder for the current learning rate value and feed the value with `feed_dict`. Possible annealing policy:
- Start with learning rate 0.0001
- Reduce learning rate 0.999 times every epoch

## Results

# References
- [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)
- [Creating Frozen Graph From Checkpoints in Tensorflow](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc)


## Other relevant resources

- https://www.superdatascience.com/opencv-face-detection/
- https://github.com/ageitgey/face_recognition
- https://github.com/deepinsight/insightface
- https://medium.freecodecamp.org/making-your-own-face-recognition-system-29a8e728107c
- https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html
- https://github.com/davidsandberg/facenet
- https://towardsdatascience.com/transfer-learning-in-tensorflow-9e4f7eae3bb4
- https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc
- https://medium.com/@RaghavPrabhu/cnn-architectures-lenet-alexnet-vgg-googlenet-and-resnet-7c81c017b848
- https://culurciello.github.io/tech/2016/06/20/training-enet.html
- https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24





