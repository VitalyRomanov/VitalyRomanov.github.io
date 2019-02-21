---
layout: post
title: Loading Tensoflow Graphs
categories: [ML]
tags: [Algorithms]
mathjax: true
show_meta: true
published: True
---

## Loading Saved Graph

```python
# assemble graph firts, names should match
with tf.Session() as sess:
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        sess.graph.as_default()

```



## Exporting Active Graph

```python
output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, # The session is used to retrieve the weights
    tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
    output_node_names # The output node names are used to select the usefull nodes
) 
with tf.gfile.GFile(output_graph_path, "wb") as f:
    f.write(output_graph_def.SerializeToString())
```



## Loading ProtoBuf Graph

```python
with tf.Session() as persisted_sess:
  with tf.gfile.GFile(protobuf_graph_path,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    # Export Graph to Tensorboard
    writer = tf.summary.FileWriter("./tf_summary", graph=persisted_sess.graph)
```

