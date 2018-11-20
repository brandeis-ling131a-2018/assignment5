# Assignment 5 - Sense Disambiguation

Due: December 7th.

(Note, there will be no assignment 6).

Word Sense Disambiguation (WSD) is the problem of identifying what sense of a word is used in a sentence, that is, what is the meaning of the word as used in the sentence. In this assignment you will build a small classifier that can determine what sense of the noun `interest` is expressed in a sentence.

One of the corpora in NLTK is the SenseEval 2 Corpus, the corpus used by the SenseEval 2 evaluation exercise for Word Sense Disambiguation. It contains about 15,000 POS-tagged and sense-tagged sentences. We will be looking at the 2368 sentences that contain the noun `interest`. You can extract all sentences using NLTK.

```python
from nltk.corpus import senseval
interest = senseval.instances('interest.pos')
```

Each element from `interest` is an instance of `SensevalInstance` which has instance variables `word`, `senses`, `position` and `context`. When you do

```python
print(instance.word, instance.position, instance.senses[0])
print(instance.context)
```

You will get

```
interest-n 18 interest_6
[('yields', 'NNS'), ('on', 'IN'), ('money-market', 'JJ'), ('mutual', 'JJ'), ('funds', 'NNS'), ('continued', 'VBD'), ('to', 'TO'), ('slide', 'VB'), (',', ','), ('amid', 'IN'), ('signs', 'VBZ'), ('that', 'IN'), ('portfolio', 'NN'), ('managers', 'NNS'), ('expect', 'VBP'), ('further', 'JJ'), ('declines', 'NNS'), ('in', 'IN'), ('interest', 'NN'), ('rates', 'NNS'), ('.', '.')]
```

For this assignment you will be editing `interest.py` and `interest.txt`. The assignment has three parts.


### 1. Analysis of senses

First get a feel of what kind of senses we are talking about. Analyze the data by finding all senses in `senseval.instances('interest.pos')`. As part of this you should use NLTK's functionality to print a  concordance and investigate what the context is for each sense. Add your findings to `interest.txt`. For each sense you should include an informal definition of the sense, anything of interest that you noted in the concordances and an example for the sense. Any code that you write for this you may add to `analysis.py`, but this is not required. We may look at the code but we will not grade it, what we grade is your write up in `interest.txt`.


### 2. Creating the classifier

This involves all the steps laid out in chapter 6 of the NLTK book: collecting data and labels, generating a function that extracts features from the data, creating training sets and test sets, and evaluating your classifier. For this assignment, you only need to use the Naive Bayes classifier, but if you feel so inclined you may experiment with other classifiers as well for a bit of extra credit. For extracting the features, note that you will extract features from an instance of  `SensevalInstance` so your function should refer to variables on that instance.

Your main focus should be on extracting features. At the least, you should collect preceding and following words and tags (recall that parts of speech are available in the Senseval data). But you should add a few more features like for example the dominating verb, or sysnset names for surrounding words, or whether there are first names in the sentence, or anything you can think of. There are no wrong features here. As part of you feature analysis you should evaluate the classifier with all features, but also try out your classifier with one or more of the features left out. Report on your results in the docstring of the module.


### 3. Running your classifier

Extract all sentences with the word 'interest' from one of the Gutenberg corpora. You could simply do this as follows:

```python
emma = nltk.corpus.gutenberg.sents('austen-emma.txt')
sents = [s for s in emma if 'interest' in s]
```

Then run your classifier on all sentences that were extracted. Note that the content of `sents` above is a list of lists. For example, the fourth element of `sents` is equal to the following:

```
['I', 'have', 'a', 'very', 'sincere', 'interest', 'in', 'Emma', '.']
```

You will need to take this sentence and create features from it, which means that you will need to POS tag it and then extract features from it. Recall that when creating the classifier you extracted features from instances of `SensevalInstance`. You can use the following code to create such an instance from tagged input:

```python
from nltk.corpus.reader.senseval import SensevalInstance

def make_instance(tagged_sentence):
    words = [t[0] for t in tagged_sentence]
    position = words.index('interest')
    return SensevalInstance('interest-n', position, tagged_sentence, [])

inst = make_instance(
            [('I', 'PRP'), ('have', 'VBP'), ('a', 'DT'), ('very', 'RB'),
             ('sincere', 'JJ'), ('interest', 'NN'), ('in', 'IN'),
             ('Emma', 'NNP'), ('.', '.')])
```

The result of this would be an object with `word`, `position`, `context` and `senses` attributes, where the value of `senses` will be an empty tuple because we do not have the sense yet.

The output of running the classifier on a sentence should be the sentence with the sense number added. For the example above the output should be

```
'I have a very sincere interest_1 in Emma .''
```


### Code layout

The code for the second and third part of the assignment should be added to `interest.py`. There is some skeleton code in that file and you could choose to follow the hints in there, but you do not have to do that. In fact, you may set up the code anyway you want and this time you are even free to use whatever function name you like.

The important thing is that your code should (1) demonstrate that you can build and evaluate the classifier and (2) show that you can run the classifier on some input from Gutenberg and print the results to the screen.

You should make it very easy for us to understand how to run your code. The docstring should specify exactly how to run your code. You may choose to set up your code in such a way that all we need to do is

```
$ python3 interest.py
```

An alternative is that you specify how to import and run your code. Code that does not run will get serious deductions.
