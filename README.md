# Assignment 5 - Sense Disambiguation

Due: December 7th.

This is a bigger assignment than all the assignments you had before so it will have a greater weight when it comes to your final grade. Note that there will be no assignment 6.

Word Sense Disambiguation (WSD) is the problem of identifying what sense of a word is used in a sentence, that is, what is the meaning of the word as used in a particular context. In this assignment you will build a small classifier that can determine what sense of the noun `interest` is expressed in a sentence. This may seem lame, but it should be relatively easy to extend your solution to this targeted problem to other nouns.

One of the corpora in NLTK is the SenseEval 2 Corpus, the corpus used by the SenseEval 2 evaluation exercise for word sense disambiguation. It contains about 15,000 POS-tagged and sense-tagged sentences. We will be looking at the 2368 sentences that contain the noun `interest`.

For this assignment you will be editing `interest.py`, `interest.txt` and `analysis.py`. The assignment has three parts.


### 1. Analysis of senses

First get a feel of what kind of senses we are talking about. Analyze the data by finding all senses in `senseval.instances('interest.pos')`. As part of this you should use NLTK's functionality to print a  concordance and investigate what the context is for each sense. Add your findings to `interest.txt`. For each sense you should include an informal definition of the sense, anything of interest that you noted in the concordances and an example for the sense. Add any code that you write for this to `analysis.py`. We may look at this code to get a handle on what you did for this analysis, but we will not grade it, what we grade is your write up in `interest.txt`.


### 2. Creating the classifier

This involves all the steps laid out in chapter 6 of the NLTK book: collecting data and labels, generating a function that extracts features from the data, creating training sets and test sets, and evaluating your classifier. For this assignment, you only need to use the Naive Bayes classifier, but if you feel so inclined you may experiment with other classifiers as well for a bit of extra credit. For extracting the features, note that you will extract features from an instance of  `SensevalInstance` so your function should refer to variables on that instance. See the hints section for more details on this.

Your main focus should be on extracting features. At the least, you should collect preceding and following words and tags (recall that parts of speech are available in the Senseval data). But you should add a few more features like for example the dominating verb, or sysnset names for surrounding words, or whether there are first names in the sentence, or anything you can think of. Hopefully, the analysis you did gives some hints as to what features would be useful. You do not need to go nuts and get many relatively hard to extract features (you can use the final project for that). Also, there are no wrong features here.

As part of you feature analysis you should evaluate the classifier with all features as well as with a subset of features by leaving one or more features out. Report on your findings in the docstring of the module.


### 3. Running your classifier

Extract all sentences with the word 'interest' from one of the Gutenberg corpora. Then run your classifier on all sentences that were extracted. You will need to take these sentences and extract features from them, which means at the least that you need to use a POS tagger. You will also want to reuse the feature extraction function you created for the previous part, which may mean that you need to create instances of `SensevalInstance`. See the hints section for code that helps you do that.


The output of running the classifier on a sentence should be the sentence with the sense number added, like the output below.

```
'I have a very sincere interest_1 in Emma .'
```

Simply write this output to the standard output (that is, the screen).


### Code layout

The code for the second and third part of the assignment should be added to `interest.py`. There is some skeleton code in that file and you could choose to follow the hints in there, but you do not have to do that. In fact, you may set up the code anyway you want and this time you are even free to use whatever function name you like.

The important thing is that your code should (1) demonstrate that you can build and evaluate the classifier and (2) show that you can run the classifier on some input from Gutenberg and print the results to the screen.

You should make it very easy for us to understand how to run your code. The docstring should specify exactly how to run your code. You may choose to set up your code in such a way that all we need to do is

```
$ python3 interest.py
```

An alternative is that you specify how to import and run your code. Code that does not run will get serious deductions.

You may use any Python module and any class or function from the `nltk` module. However, we do not want to install anything so your code should not rely on other third-party modules.



### Hints and some background

**Getting your input**

Under the hood, the source of the Senseval data that we use lives in the file `interest.pos` in `~/nltk_data/corpora/senseval/` and looks as follows:

```xml
<instance id="interest-n.int1">

<answer instance="interest-n.int1" senseid="interest_6"/>

<context>
<wf pos="NNS">yields</wf> <wf pos="IN">on</wf> <wf pos="JJ">money-market</wf>
<wf pos="JJ">mutual</wf> <wf pos="NNS">funds</wf> <wf pos="VBD">continued</wf>
<wf pos="TO">to</wf> <wf pos="VB">slide</wf> <wf pos=",">,</wf>
<wf pos="IN">amid</wf> <wf pos="VBZ">signs</wf> <wf pos="IN">that</wf>
<wf pos="NN">portfolio</wf> <wf pos="NNS">managers</wf> <wf pos="VBP">expect</wf>
<wf pos="JJ">further</wf> <wf pos="NNS">declines</wf> <wf pos="IN">in</wf> <head>
<wf pos="NN">interest</wf></head> <wf pos="NNS">rates</wf> <wf pos=".">.</wf>
</context>

</instance>
```

You can extract all sentences using one of NLTK's corpus readers:

```python
from nltk.corpus import senseval
interest = senseval.instances('interest.pos')
```

Here, `senseval` is in instance of `SensevalCorpusReader` which has a special method `instances` that returns an object that acts like a sequence, that is, you can access indices and slices and you can loop over it. Each element from `interest` is an instance of `SensevalInstance` which has instance variables `word`, `senses`, `position` and `context`. When you do

```python
instance = interest[22]
print(instance.word, instance.position, instance.senses[0])
print(instance.context)
```

You will get

```
interest-n 12 interest_3
[('``', '``'), ('we', 'PRP'), ('believe', 'VBP'), ('that', 'IN'), ('it', 'PRP'),
('is', 'VBZ'), ('vitally', 'RB'), ('important', 'JJ'), ('for', 'IN'),
('those', 'DT'), ('japanese', 'NN'), ('business', 'NN'), ('interests', 'NNS'),
('{', '('), ('in', 'IN'), ('the', 'DT'), ('u', 'PRP'), ('.', '.'), ('s', 'PRP'),
('.', '.')]
```

**Getting data for running the classifier**

To extract all sentences with `interest` from Jame Austen's Emma you could simply do the following:

```python
emma = nltk.corpus.gutenberg.sents('austen-emma.txt')
sents = [s for s in emma if 'interest' in s]
```

Note that the content of `sents` above is a list of lists. For example, the fourth element of `sents` is equal to the following:

```
['I', 'have', 'a', 'very', 'sincere', 'interest', 'in', 'Emma', '.']
```

** Creating instances of SensevalInstance**

When creating the classifier you probably extracted features from instances of `SensevalInstance`. You can use the following code to create such an instance from tagged input:

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

The result of this is an object with `word`, `position`, `context` and `senses` attributes, where the value of `senses` will be an empty tuple because we do not have the sense yet.
