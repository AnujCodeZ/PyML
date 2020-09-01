# Vectors
from typing import List, Tuple, Callable
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

# Matrices
Matrix = List[List[float]]

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns shape of matrix A"""
    num_rows = len(A)
    num_columns = len(A[0]) if A else 0
    return num_rows, num_columns

def get_row(A: Matrix, i: int) -> Vector:
    """Returns i-th row of A as a Vector"""
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """Returns j-th column of A as a Vector"""
    return [A_i[j]
            for A_i in A]

def make_matrix(num_rows: int, num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """Returns a num_rows x num_cols matrix
    whose (i, j)-th entry is entry-fn(i, j)
    """
    return [[entry_fn(i, j)
             for j in range(num_cols)]
            for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
    """Returns n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)