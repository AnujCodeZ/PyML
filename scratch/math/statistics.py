# Central Tendencies
from typing import List
from collections import Counter

def mean(xs: List[float]) -> float:
    """Returns the mean of xs"""
    return sum(xs) / len(xs)

def median(xs: List[float]) -> float:
    """Returns the median of xs"""
    sorted_xs = sorted(xs)
    if len(xs) % 2 == 0:
        hi = len(xs) // 2
        return (sorted_xs[hi-1] + sorted_xs[hi]) / 2 
    else:
        return sorted_xs[len(xs) // 2]

def quantile(xs: List[float], p: float) -> float:
    """Returns the pth-percentile value of xs"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

def mode(xs: List[float]) -> List[float]:
    """Returns a list of mode values of xs"""
    counts = Counter(xs)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]