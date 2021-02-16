import numpy as np
import numpy.linalg as la
import pytest
import cvxpy as cvx
import itertools

def clique_matrix( k):
    a = np.zeros( (k-1, k*(k-1)//2))
    for idx, (n0,n1) in enumerate(itertools.combinations( range(k), 2)):
        if n0 > 0:
            a[n0-1][idx] =  1
        if n1 > 0:
            a[n1-1][idx] = -1
    return a

def solve_l1( k, a, source):
    x = cvx.Variable(a.shape[1])
    objective = cvx.Minimize(cvx.norm(x,1))
    constraints = [a @ x == source[:k-1]]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    print( f'l1opt {print_float_array(np.array(x.value))}')

def solve_all( k, a, source):
    assert a.shape[0] + a.shape[1] == source.shape[0]

    r = .001
    b = np.block( [[np.zeros((a.shape[0],a.shape[0])),a],[a.T, np.identity(a.shape[1])*-r]])

    gram = a.dot(a.T)
    print( f'source {source}')
    print( f'resistors {print_float_array(la.inv(b).dot(source)[k-1:])}')
    print( f'l2opt {print_float_array(a.T.dot(la.inv(gram).dot( source[:k-1])))}')
    solve_l1( k, a, source)

def test_clique_matrix():
    assert (2, 3) == clique_matrix( 3).shape
    assert (3, 6) == clique_matrix( 4).shape
    assert (4, 10) == clique_matrix( 5).shape

def print_float_array( a):
    lst = [ f'{x:.1f}' for x in a]
    return f"[{' '.join(lst)}]"

def test_A3():
    k = 3
    a = clique_matrix(k)

    source = np.array([6,3,0,0,0])
    solve_all( k, a, source)

    source = np.array([1,1,0,0,0])
    solve_all( k, a, source)

    source = np.array([3,-3,0,0,0])
    solve_all( k, a, source)

def test_A4():
    k = 4
    a = clique_matrix( k)

    source = np.array([1,1,1,0,0,0,0,0,0])
    solve_all( k, a, source)

    source = np.array([8,4,0,0,0,0,0,0,0])
    solve_all( k, a, source)
