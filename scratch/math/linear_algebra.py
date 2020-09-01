# Vectors
from typing import List
import math

Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements of two vectors."""
    assert len(v) == len(w), "vectors must be the same length"
    
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements of two vectors."""
    assert len(v) == len(w), "vectors must be the same length"
    
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements in a list of vectors"""
    assert vectors, "no vectors provided"
    n = len(vectors[0])
    assert all(len(v) == n for v in vectors), "different sizes!"
    
    return [sum(vector[i] for vector in vectors)
            for i in range(n)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiply every element of vector by c"""
    return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average of list of vectors"""
    n = len(vectors)
    
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"
    
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def magnitude(v: Vector) -> float:
    """Returns magnitude (or length) of v"""
    return math.sqrt(dot(v, v))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return magnitude(subtract(v, w))