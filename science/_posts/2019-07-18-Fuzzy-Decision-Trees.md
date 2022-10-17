---
layout: post
title: Fuzzy Decision Trees
categories: [science-papers]
tags: [Fuzzy Logic]
mathjax: true
show_meta: true
published: false
---

Fuzzy Decision Trees is a type of learning algorithm based on Fuzzy Logos. The main idea is similar to the regular decision trees. The difference lies in the way we propagate a decision over the tree. In Fuzzy Logic, we evaluate how well an object associates with some fuzzy set. Every fuzzy set is associated with a function that takes the parameters of an object as an input and returns the degree to which this object belongs to this fuzzy set. 

Fuzzy decision trees work i na similar way. Every node in the tree can be seen as a fuzzy set. Every child node is a subset of the parent fuzzy set. The root node is some sort of a superset that covers all of the subsets under consideration.

Consider a binary decision tree. At every node we evaluate the degree of an object belonging to the current fuzzy set. When we evaluate the two children of the current node, we will multiply this degree by the degrees calculated in children nodes. Thus, by the time we reach leaves of the decision trees we will have some sort of confidence of how well an object belongs to the current class. But this is not where the benefits of the fuzzy decision process ends. 

In decision trees, it is possible that we arrive to the same conclusion via different paths. Fuzzy decision trees allwo to aggregate this decision via the process of defuzzification.  



Genetically optimized FDT

The idea of the method is similar to other FDT approaches. First we train a C4.5 tree, and adjust its parameters later. Classical decision tree method creates a set of rules with a strict decision boundary. This approach relaxes the decision rule to a fuzzy decision rule and then optimized the parameters of a fuzzy membership function. The Genetic algorithm was chosen for couple of reasons. First, the backpropagation when the tree is very deep seems to be a bad idea due to a vanishing gradients. Also, the gradients depend highly on the form of the membership function selected for each node. In contrast, the GA allows to find the global optimum. The loss in this case is simply the LS error.

The final decision of FDT is created by the means of s or t norm, or the defuzzification rule in the case of continuous output.