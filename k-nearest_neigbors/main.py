import numpy as np
from collections import Counter


def majority(labels):
    counts = Counter(labels)
    most, most_count = counts.most_common(1)[0]
    num_of_most = len([count
                       for count in counts.values()
                       if count == most_count])
    if num_of_most == 1:
        return most
    else:
        return majority(labels[:-1])

def distance(p1, p2):
    sub = np.subtract(p1 - p2)
    return np.sqrt(np.dot(sub, sub))

class LabeledPoint:
    def __init__(self, point, label):
        self.point = point
        self.label = label

def knn_classify(k, labeled_points, new_point):
    
    by_distance = sorted(labeled_points,
                         key=lambda lp: distance(lp.point, new_point))
    k_nearest_labels = [lb.label for lp in by_distance[:k]]
    return majority(k_nearest_labels)