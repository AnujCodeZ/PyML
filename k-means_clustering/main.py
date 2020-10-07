import tqdm
import random
import itertools
import numpy as np


def num_differences(v1, v2):
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])

def cluster_means(k, inputs, assignments):
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)
    
    return [np.mean(cluster) if cluster else random.choice(inputs)
            for cluster in clusters]

class KMeans:
    def __init__(self, k):
        self.k = k
        self.means = None
    
    def classify(self, input):
        return min(range(self.k),
                   key=lambda i: np.dot(np.subtract(input, self.means[i]),
                                        np.subtract(input, self.means[i])))
    
    def train(self, inputs):
        assignments = [random.randrange(self.k) for _ in inputs]
        
        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]
                num_changed = num_differences(assignments, new_assignments)
                if num_changed == 0:
                    return
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f'changed: {num_changed} / {len(inputs)}')
        