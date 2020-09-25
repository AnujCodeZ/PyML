import numpy as np
from collections import Counter, defaultdict


def entropy(labels):
    total = len(labels)
    ps = [count / total for count in Counter(labels).values()]
    return sum(-p * np.log2(p)
               for p in ps
               if p > 0)


def partition(inputs, feature):
    partitions = defaultdict(list)
    for input in inputs:
        key = getattr(input, feature)
        partitions[key].append(input)
    return partitions


def partition_entropy(inputs, feature, label):
    partitions = partition(inputs, feature)

    labels = [[getattr(input, label)
               for input in partition]
              for partition in partitions.values()]

    total = sum(len(label) for label in labels)
    return sum(entropy(label) * len(label) / total
                  for label in labels)

class Leaf:

    def __init__(self, value):
        self.value = value


class Node:

    def __init__(self, feature, subtrees, default_value=None):
        self.feature = feature
        self.subtrees = subtrees
        self.default_value = default_value


class Tree:

    def train(self, inputs, features, label):
        label_counts = Counter(getattr(input, label)
                               for input in inputs)
        most_common_label = label_counts.most_common(1)[0][0]

        if len(label_counts) == 1 and not features:
            return Leaf(most_common_label)

        def split_entropy(feature):
            return partition_entropy(inputs, feature, label)

        best_feature = min(features, key=split_entropy)

        partitions = partition(inputs, best_feature)
        new_features = [a for a in features if a != best_feature]

        subtrees = {feature_value: self.train(subset,
                                              new_features,
                                              label)
                    for feature_value, subset in partitions.items()}

        return Node(best_feature, subtrees, default_value=most_common_label)

    def predict(self, tree, input):

        if isinstance(tree, Leaf):
            return tree.value

        subtree_key = getattr(input, tree.feature)

        if subtree_key not in tree.subtrees:
            return tree.default_value

        subtree = tree.subtrees[subtree_key]
        return self.predict(subtree, input)

# Example


class Candidate:

    def __init__(self, level, lang, tweets, phd, did_well=None):
        self.level = level
        self.lang = lang
        self.tweets = tweets
        self.phd = phd
        self.did_well = did_well


inputs = [Candidate('Senior', 'Java', False, False, False),
          Candidate('Senior', 'Java', False, True, False),
          Candidate('Mid', 'Python', False, False, True),
          Candidate('Junior', 'Python', False, True, False),
          Candidate('Junior', 'R', True, False, True),
          Candidate('Junior', 'R', True, True, False),
          Candidate('Mid', 'R', True, True, True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R', True, False, True),
          Candidate('Junior', 'Python', True, False, True)
          ]

model = Tree()

tree = model.train(inputs,
                   ['level', 'lang', 'tweets', 'phd'],
                   'did_well')

print(model.predict(tree, Candidate('Junior', 'Python', True, False)))