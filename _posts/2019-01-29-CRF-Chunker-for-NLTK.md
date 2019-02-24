---
layout: post
title: CRF Chunker for NLTK
categories: [NLP]
tags: [Algorithms]
mathjax: true
show_meta: true
published: True
---

The cournerstone of any natural language understanding system is NLP algorithms. The classical set of algorithms constitutes an NLP ptocessing pipeline:

**Tokenization -> POS Tagging -> Chunking -> NER -> Syntactic Parsing**

It is hard to do anything meaningful without this pipeline set up when processing text. Several APIs from large companies are available, but we want open source, right? While pursuing the goal of assembling the NLP pipeline for russian language, I found issues with existing solutions for everything further than POS tagging. To be fair, even POS taggers are not flawless. Let's work on this problem incrementally and address the issue of chunking first. Further, a way to create a CRF chunker with NLTK is discussed, and its performance on russian language is compared with other algorithms in NLTK library.  

The outline of the article is the following 

{:toc}

## NLTK overview

NLTK is a mature library for text processing. Fortunately for us, it comes with an extensive [usage guide](https://www.nltk.org/book/) that describes ins and outs of text processing with NLTK.

**Tokenization**

NLTK already has rule-based tokenizer for words `word_tokenizer`, tokenizer for sentences `sent_tokenize`, and even an [unsupervised algorithm](https://www.nltk.org/_modules/nltk/tokenize/punkt.html) for learning sentence tokenization rules.

**POS Tagging**

There are statistical POS tokenizers for english and russian languages available, you only need to download a pretrained model. Refer to the documentation of [`pos_tag` method](https://www.nltk.org/_modules/nltk/tag.html).

**Chunking and NER**

The last two steps of the pipeline, namely chunking and NER are a little bit harder to implement, but essentially no different from tagging algorithms. There are no models available.

**Syntactinc Parsing**

To the best of my knowledge, not proper syntactic parsing in NLTK.

**How to implement chunker with NLTK?**

NLTK has several [algorithms for chunking](http://www.nltk.org/howto/chunk.html) implemented. The most common is [grammar based chunker](https://www.nltk.org/_modules/nltk/chunk/regexp.html), but there is also a [statistical chunker](https://www.nltk.org/_modules/nltk/chunk/named_entity.html) model that requires training. Those who are interested can have a look inside [corresponsing source directory](https://github.com/nltk/nltk/tree/develop/nltk/chunk).

I was mostly interested in the statistical model, since, as I discovered, creating a proper grammar-based chunker requires a great deal of linguistic profficiency and work. Insted I desided to use a shortcut - machine learning. It is ideal when you do not want to elaborately enumerate all the rules for proper classification. And chunking is nothing more than a classification task where the input is the word, its neighbours, and their corresponding POS tags.

The class for statistical chunker in NLTK is `NEChunkParser`. You can see that it has abbreviation for Named Entity in its name, hinting that we can use the same class for both tasks of chunking and NER. Just need to provide relevant data. 

By default `NEChunkParser`  uses `megam` backend which proved to be hard to instal on non-linux machine. Also, I have heard that state of the state-of-the-art models are based on Conditional Random Fields (CRF), and NLTK conveniently has a [CRF tagger](https://www.nltk.org/_modules/nltk/tag/crf.html).

So instead of using already available class I desided to create one more chunker based on CRF tagger. Before going further, let's discuss data formats for chunkers.

## IOB and BILUO tags

Chunking can be considered as a classification task where we neet to estimate the label for every token in a sentence. The labels depends on a selected labeling scheme. Below is an example of IOB tags

![](https://www.nltk.org/images/chunk-tagrep.png)

*Source: [NLTK book](https://www.nltk.org/book/ch07.html)*

IOB tags are represented by B-, I- and O prefixes for the chunk tags. The example above shows tags for Noun Phrases (NP), but chunking into verb phrases is also common. The meaning of tags is the following

| Tag  | Desctiption                                                  |
| ---- | ------------------------------------------------------------ |
| B-\*  | Token is the beginning of the chunk. One work chunks are also labeled with B-\* tags |
| I-\*  | Token is inside or in the end of a chunk                     |
| O    | Token is not a part of any chunk                             |

The problem with this labeling scheme is that it is very biased towards B-\* tags, especially when many one word chunks are present. A better labeling scheme is BILUO tags

| Tag  | Desctiption                          |
| ---- | ------------------------------------ |
| B-\* | Token is the beginning of the chunk. |
| I-\* | Token is inside of a chunk           |
| L-\* | Token is the last in the chunk       |
| U-\* | Token is a single-word chunk         |
| O    | Token is not a part of any chunk     |

For the example above, the data can be represented as a list of tuples

```python
sentence = [
    (We, PRP, U-NP),
    (saw, VBD, O),
    (the, DET, B-NP),
    (yellow, JJ, I-NP),
    (dog, NN, L-NP)
]
```

Both of these two labeling scemes contain the same amount of information and are easily converted one from another. Unfortunately, NLTK has proper support only for IOB tags, so if we want to use BILUO for training we need to do conversion to BILUO and back to IOB under the hood of a chunker. 

Now, since we are using a `CRFTagger` for the chunker, we need to pay attention to the differences between interfaces for chunkers and taggers. Ideally, we need to implement an instance of chunker that will be a wrapper for a tagger. Let's look at the differences between interfaces. 

## Inputs for `CRFTagger`

The beauty of open sorce - you can learn how everything works. The implementation of `CRFTagger` is [publicly available](https://github.com/nltk/nltk/blob/develop/nltk/tag/crf.py). From the source code we can see, that the tagger accepts the list of tuples as training input. The format of tuples is `(word, pos_tag)`, but the type of the word does not have to be string. It could contain any structure, the only requirement is that you also provide a featurization function that will prepare input features for the CRF classifier. 

Since we solve the problem of chunking, and POS tags are a valuable inputs. Other NLTK chunking algorithm accept tuples `(word, pos_tag, iob_tag)` as the training input.  Existing CRFTagger will work fine with the trainig input with the format `((word, pos_tags), iob_tag)`. I will need to provide a featurization function for this custom input format. Let's borrow it from `NEChunkParser` with slight modifications

```python
def _feature_detector(self, tokens, index):
        def shape(word):
            if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word, re.UNICODE):
                return 'number'
            elif re.match('\W+$', word, re.UNICODE):
                return 'punct'
            elif re.match('\w+$', word, re.UNICODE):
                if word.istitle():
                    return 'upcase'
                elif word.islower():
                    return 'downcase'
                else:
                    return 'mixedcase'
            else:
                return 'other'


        def simplify_pos(s):
            if s.startswith('V'):
                return "V"
            else:
                return s.split('-')[0]

        word = tokens[index][0]
        pos = simplify_pos(tokens[index][1])
        if index == 0:
            prevword = prevprevword = ""
            prevpos = prevprevpos = ""
            prevshape = prevtag = prevprevtag = ""
        elif index == 1:
            prevword = tokens[index - 1][0].lower()
            prevprevword = ""
            prevpos = simplify_pos(tokens[index - 1][1])
            prevprevpos = ""
            prevtag = "" 
            prevshape = prevprevtag = ""
        else:
            prevword = tokens[index - 1][0].lower()
            prevprevword = tokens[index - 2][0].lower()
            prevpos = simplify_pos(tokens[index - 1][1])
            prevprevpos = simplify_pos(tokens[index - 2][1])
            prevtag = "" 
            prevprevtag = "" 
            prevshape = shape(prevword)
        if index == len(tokens) - 1:
            nextword = nextnextword = ""
            nextpos = nextnextpos = ""
        elif index == len(tokens) - 2:
            nextword = tokens[index + 1][0].lower()
            nextpos = tokens[index + 1][1].lower()
            nextnextword = ""
            nextnextpos = ""
        else:
            nextword = tokens[index + 1][0].lower()
            nextpos = tokens[index + 1][1].lower()
            nextnextword = tokens[index + 2][0].lower()
            nextnextpos = tokens[index + 2][1].lower()

        features = {
            'shape': '{}'.format(shape(word)),
            'wordlen': '{}'.format(len(word)),
            'prefix3': word[:3].lower(),
            'suffix3': word[-3:].lower(),
            'pos': pos,
            'word': word,
            'prevpos': '{}'.format(prevpos),
            'nextpos': '{}'.format(nextpos),
            'prevword': '{}'.format(prevword),
            'nextword': '{}'.format(nextword),
            'prevprevword': '{}'.format(prevprevword),
            'nextnextword': '{}'.format(nextnextword),
            'word+nextpos': '{0}+{1}'.format(word.lower(), nextpos),
            'word+nextnextpos': '{0}+{1}'.format(word.lower(), nextnextpos),
            'word+prevpos': '{0}+{1}'.format(word.lower(), prevpos),
            'word+prevprevpos': '{0}+{1}'.format(word.lower(), prevprevpos),
            'pos+nextpos': '{0}+{1}'.format(pos, nextpos),
            'pos+nextnextpos': '{0}+{1}'.format(pos, nextnextpos),
            'pos+prevpos': '{0}+{1}'.format(pos, prevpos),
            'pos+prevprevpos': '{0}+{1}'.format(pos, prevprevpos),
        }
```

## Assembling `CRFChunkParser`

Chunkers accept the training tata in the format `(word, pos_tag, iob_tag)`. Let's keep it this way for compatibility with other classes and rather transforma the data in the suitable format before training.

```python
def triplets2tagged_pairs(iob_sent):
     return [((word, pos), chunk) for word, pos, chunk in iob_sent]
chunked_sents = [triplets2tagged_pairs(sent) for sent in chunked_sents]
```

Next, we are aready to feed the data to `CRFTagger`. 

```python
tagger = CRFTagger(feature_func=_feature_detector, training_opt=training_opt)
tagger.train(chunked_sents, model_file)
```

where `training_opt` are additional training parameters, and `model_file` is the path where CRF model will be stored. After assembling everything we get the following class

```python
class CRFChunkParser(ChunkParserI):
    def __init__(self, chunked_sents=[], feature_func=None, model_file=None, training_opt={}):
 
        # Transform the trees in IOB annotated sentences [(word, pos, chunk), ...]
        # chunked_sents = [tree2conlltags(sent) for sent in chunked_sents]
 
        # Transform the triplets in pairs, make it compatible with the tagger interface [((word, pos), chunk), ...]
        def triplets2tagged_pairs(iob_sent):
            return [((word, pos), chunk) for word, pos, chunk in iob_sent]
        chunked_sents = [triplets2tagged_pairs(sent) for sent in chunked_sents]
 
        if feature_func is not None:
            feat_func = feature_func
        else:
            feat_func = self._feature_detector
        self.tagger = CRFTagger(feature_func=feat_func, training_opt=training_opt)
        if not model_file:
            raise Exception("Provide path to save model file")
        self.model_file = model_file
        if chunked_sents:
            self.train(chunked_sents)
        else:
            self.tagger.set_model_file(self.model_file)

    def train(self, chunked_sents):
        self.tagger.train(chunked_sents, self.model_file)
    
    def load(self, model_file):
        self.tagger.set_model_file(model_file)
 
    def parse(self, tagged_sent, return_tree = True):
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets) if return_tree else iob_triplets


    def _feature_detector(self, tokens, index):
        ... # See the listing above
```



## Comparing `RegexpParser`, `NEChunkParser` and `CRFChunkParser` for russian language

### Only Noun Phrases

#### `RegexpParser`

Assume `RegexpParser` has simple grammar for detecting noun chunks

```python
from nltk import RegexpParser
grammar = r"""
NP:
{<S.*|A.*>*<S.*>}  # Nouns and Adjectives, terminated with Nouns
"""
chunker = RegexpParser(grammar)
```

The performance of such chunker obrained with `chunker.evaluate()`

```
ChunkParse score:
    IOB Accuracy:  82.9%%
    Precision:     62.1%%
    Recall:        43.7%%
    F-Measure:     51.3%%
```

#### `CRFChunkParser` with `IOB` scheme

```
ChunkParse score:
    IOB Accuracy:  95.5%%
    Precision:     89.5%%
    Recall:        90.2%%
    F-Measure:     89.9%%
```

#### `CRFChunkParser` with `BILUO` scheme

```
ChunkParse score:
    IOB Accuracy:  96.0%%
    Precision:     91.9%%
    Recall:        91.9%%
    F-Measure:     91.9%%

```

### Full Chunker

Full chunker groups tokens into noun, verb and prepositional chunks.

#### `CRFChunkParser` with `BILUO` scheme and extra features

The original set of features was a little off in a way that we often provided the whole word as a feature. Evidently, this will make it harder for the algorithm to generalize. Instead, we try to look on the prefix and suffix of the word to understand the dependencies between words. 

```python
def _feature_detector(self, tokens, index):
    def shape(word):
        if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word, re.UNICODE):
            return 'number'
        elif re.match('\W+$', word, re.UNICODE):
            return 'punct'
        elif re.match('\w+$', word, re.UNICODE):
            if word.istitle():
                return 'upcase'
            elif word.islower():
                return 'downcase'
            else:
                return 'mixedcase'
        else:
            return 'other'


    word = tokens[index][0]
    pos = tokens[index][1]

    if index == 0:
        prevword = prevprevword = ""
        prevpos = prevprevpos = ""
        prevshape = ""
    elif index == 1:
        prevword = tokens[index - 1][0].lower()
        prevprevword = ""
        prevpos = tokens[index - 1][1]
        prevprevpos = ""
        prevshape = ""
    else:
        prevword = tokens[index - 1][0].lower()
        prevprevword = tokens[index - 2][0].lower()
        prevpos = tokens[index - 1][1]
        prevprevpos = tokens[index - 2][1]
        prevshape = shape(prevword)
    if index == len(tokens) - 1:
        nextword = nextnextword = ""
        nextpos = nextnextpos = ""
    elif index == len(tokens) - 2:
        nextword = tokens[index + 1][0].lower()
        nextpos = tokens[index + 1][1].lower()
        nextnextword = ""
        nextnextpos = ""
    else:
        nextword = tokens[index + 1][0].lower()
        nextpos = tokens[index + 1][1].lower()
        nextnextword = tokens[index + 2][0].lower()
        nextnextpos = tokens[index + 2][1].lower()

    def get_suffix_prefix(wordm, length):
        if len(word)>length:
            pref = word[:length].lower()
            suf = word[-length:].lower()
        else:
            pref = word
            suf = ""
        return pref, suf

    suf_pref_lengths = [2,3]
    words = {
        'word': {'w': word, 'pos': pos, 'shape': shape(word)},
        'nword': {'w': nextword, 'pos': nextpos, 'shape': shape(nextword)},
        'nnword': {'w': nextnextword, 'pos': nextnextpos, 'shape': shape(nextnextword)},
        'pword': {'w': prevword, 'pos': prevpos, 'shape': shape(prevprevword)},
        'ppword': {'w': prevprevword, 'pos': prevprevpos, 'shape': shape(prevprevword)}
    }

    base_features = {}
    for word_position in words:
        for item in words[word_position]:
            if item == 'w': continue
            base_features[word_position+"."+item] = words[word_position][item]

    prefix_suffix_features = {}
    for word_position in words:
        for l in suf_pref_lengths:
            feature_name_base = word_position+"."+repr(l)+"."
            pref, suf = get_suffix_prefix(words[word_position]['w'], l)
            prefix_suffix_features[feature_name_base+'pref'] = pref
            prefix_suffix_features[feature_name_base+'suf'] = suf
            prefix_suffix_features[feature_name_base+'pref.suf'] = '{}+{}'.format(pref, suf)
            prefix_suffix_features[feature_name_base+'posfix'] = '{}+{}+{}'.format(pref, words[word_position]['pos'], suf)
            prefix_suffix_features[feature_name_base+'shapefix'] = '{}+{}+{}'.format(pref, words[word_position]['shape'], suf)

    features = {

        'pos': pos,
        'prevpos': prevpos,
        'nextpos': nextpos,
        'prevprevpos': prevprevpos,
        'nextnextpos': nextnextpos,
        
        'pos+nextpos': '{0}+{1}'.format(pos, nextpos),
        'pos+nextnextpos': '{0}+{1}'.format(pos, nextnextpos),
        'pos+prevpos': '{0}+{1}'.format(pos, prevpos),
        'pos+prevprevpos': '{0}+{1}'.format(pos, prevprevpos),
    }

    features.update(base_features)
    features.update(prefix_suffix_features)

    return features
```

Achieved performance is

```
ChunkParse score:
    IOB Accuracy:  95.4%%
    Precision:     93.5%%
    Recall:        93.7%%
    F-Measure:     93.6%%
```

## References

https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb