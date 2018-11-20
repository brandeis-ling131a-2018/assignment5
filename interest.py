"""

Skeleton code. Note that this is not valid code due to all the dots.

"""

import random

import nltk
from nltk.corpus import senseval
from nltk.corpus.reader.senseval import SensevalInstance


def create_labeled_data():
    # collect all data from the corpus
    interest = senseval.instances('interest.pos')
    # create labeled data
    labeled_data = ...
    return labeled_data


def create_feature_sets(labeled_data):
    # create feature sets
    ...
    return train_set, test_set


def wsd_features(instance):
    ...
    return { ... }


def make_instance(tagged_sentence):
    words = [t[0] for t in tagged_sentence]
    position = words.index('interest')
    return SensevalInstance('interest-n', position, tagged_sentence, [])


def train_classifier(training_set):
    # create the classifier
    ...


def evaluate_classifier(classifier, test_set):
    # get the accuracy and print it
    ...


def run_classifier(classifier):
    ...


if __name__ == '__main__':

    labeled_data = create_labeled_data()
    training_set, test_set = create_feature_sets(labeled_data)
    classifier = train_classifier(training_set)
    evaluate_classifier(classifier, test_set)
    run_classifier(classifier)
