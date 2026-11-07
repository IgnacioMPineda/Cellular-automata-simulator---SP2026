import numpy as np
import itertools
import random
import matplotlib.pyplot as plt


from Functions_for_Notebook_1 import make_rule, make_rule_zero_condition, make_rule_symmetry #For making the rules

from Functions_for_Notebook_1 import dynamics, dynamics_reflection, dynamics_periodic #For the dynamics

from Functions_for_Notebook_1 import plot_grid #For the plotting

from Functions_for_Notebook_1 import complement_rule
# From Notebook 2
from Functions_for_Notebook_2 import plot_grid_multiple, plot_multiple_cells, dynamics_initialandfinal #To grid multiple systems

from Functions_for_Notebook_2 import is_rule

from Functions_for_Notebook_2 import is_reversible, gardens_of_eden



# From Notebook 3
from Functions_for_Notebook_3 import vectorized_ic, next_state, markov_matrix, markov_matrix_mapping  # Describe evolutions with matrices

from Functions_for_Notebook_3 import probabilistic_dynamics, probabilistic_dynamics_initialandfinal

from Functions_for_Notebook_3 import semiclassical_dynamics, semiclassical_markov_matrix # For probabilistic systems


def is_normal(M):
    #M M^T = M^T M.
    return np.array_equal(M @ M.T, M.T @ M)

def is_doubly_stochastic(M):
    #Non-negative, columns and rows all sum to 1.
    return (np.all(M >= 0) and np.allclose(M.sum(axis=0), 1)
            and np.allclose(M.sum(axis=1), 1))

def is_permutation(M):
    #Entries 0/1, exactly one 1 in every row and every column.
    return (np.all((M == 0) | (M == 1))
            and np.all(M.sum(axis=0) == 1) and np.all(M.sum(axis=1) == 1))

def is_bijective(M):
    #The successor map f is one-to-one (no two states share a successor).
    f = M.argmax(axis=0)
    return len(np.unique(f)) == len(f)

def numerical_check(M, rtol=1e-6, atol=1e-6):
    """Every |lambda| is close to 1 (relative tolerance) or close to 0 (absolute tolerance)."""
    mods = np.abs(np.linalg.eigvals(M.astype(float)))
    near_one  = np.isclose(mods, 1, rtol=rtol, atol=0)
    near_zero = np.isclose(mods, 0, rtol=0, atol=atol)
    return bool(np.all(near_one | near_zero)), mods

def preimage_counts(M):
    return M.sum(axis=1).astype(int)         

def singular_values_det(M):
    return np.sqrt(preimage_counts(M))

def garden_of_eden(M):
    return np.where(preimage_counts(M) == 0)[0]

def successor_array(rule, L):
    n = 1 << L
    w = 1 << np.arange(L - 1, -1, -1)
    s = (np.arange(n)[:, None] & w) > 0                    
    p = np.pad(s.astype(int), ((0, 0), (1, 1)))            # fixed 0 boundaries
    table = (rule >> np.arange(8)) & 1
    nxt = table[4 * p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]]
    return nxt @ w

def transfer_matrices(rule):
    """T[b][(a,m),(m,c)] = 1 if f(a,m,c) = b. Nodes = neighbour pairs (x_i, x_{i+1})."""
    T = np.zeros((2, 4, 4), dtype=object)
    for a in (0, 1):
        for m in (0, 1):
            for c in (0, 1):
                out = (rule >> (4*a + 2*m + c)) & 1
                T[out, 2*a + m, 2*m + c] = 1
    return T

def is_reversible(rule, L):
    """Make the boundaries zero"""
    T = transfer_matrices(rule)
    P = np.kron(T[0], T[0]) + np.kron(T[1], T[1])      
    u = np.array([1, 1, 0, 0], dtype=object)           
    v = np.array([1, 0, 1, 0], dtype=object)            
    R, B, k = np.eye(16, dtype=object), P.copy(), L
    while k:
        if k & 1:
            R = R.dot(B)
        B = B.dot(B)
        k >>= 1
    return np.kron(u, u).dot(R).dot(np.kron(v, v)) == 2**L

def bijective_lengths(rule, Lmax):
    T = transfer_matrices(rule)
    P = np.kron(T[0], T[0]) + np.kron(T[1], T[1])
    w   = np.kron([1, 1, 0, 0], [1, 1, 0, 0]).astype(object)   
    end = np.kron([1, 0, 1, 0], [1, 0, 1, 0]).astype(object)   
    out = np.empty(Lmax, dtype=bool)
    for L in range(1, Lmax + 1):
        w = w.dot(P)
        out[L - 1] = w.dot(end) == 2**L
    return out

def N(L):
    return (25/3)+np.cos(np.pi*L) - (4/3)*np.cos( (2*np.pi*(L-2)) / 3)
