import math
from typing import List
from collections import Counter
from linear_algebra import dot


# Central Tendencies

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

# Dispersion

def data_range(xs: List[float]) -> float:
    """Returns range of the x"""
    return max(xs) - min(xs)


def de_mean(xs: List[float]) -> List[float]:
    """Translate xs by subtracting its mean"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def variance(xs: List[float]) -> float:
    """Almost the average squared deviation from the mean"""
    assert len(xs) >= 2, "Variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    return dot(xs, xs) / (n-1)


def standard_deviation(xs: List[float]) -> float:
    """The standard deviation is the square root of variance of x"""
    return math.sqrt(variance(xs))


def interquartile_range(xs: List[float]) -> float:
    """Returns the difference between the 75%-ile and the 25%-ile"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)

# Correlation

def covariance(xs: List[float], ys: List[float]) -> float:
    """Returns covariance of x and y"""
    assert len(xs) == len(ys), "xs and ys must have same number of elements"
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

def correlation(xs: List[float], ys: List[float]) -> float:
    """Measures how much xs and ys vary in tendem about thier means"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0