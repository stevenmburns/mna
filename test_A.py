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

def solve_l1( a, source):
    x = cvx.Variable(a.shape[1])
    objective = cvx.Minimize(cvx.norm(x,1))
    constraints = [a @ x == source[:a.shape[0]]]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    print( f'l1opt {print_float_array(np.array(x.value))}')

def solve_all( a, currents):
    print()

    assert a.shape[0] == currents.shape[0]

    source = np.block( [currents, np.zeros( (a.shape[1],))])

    assert a.shape[0] + a.shape[1] == source.shape[0]

    r = .001
    b = np.block( [[np.zeros((a.shape[0],a.shape[0])),a],[a.T, np.identity(a.shape[1])*-r]])

    gram = a.dot(a.T)
    print( f'source {print_float_array(source)}')
    print( f'resistors {print_float_array(la.inv(b).dot(source)[a.shape[0]:])}')
    print( f'l2opt a{print_float_array(a.T.dot(la.inv(gram).dot( source[:a.shape[0]])))}')
    solve_l1( a, source)

@pytest.mark.skip
def test_clique_matrix():
    assert (2, 3) == clique_matrix( 3).shape
    assert (3, 6) == clique_matrix( 4).shape
    assert (4, 10) == clique_matrix( 5).shape

def print_float_array( a):
    lst = [ f'{x:.1f}' for x in a]
    return f"[{' '.join(lst)}]"

#@pytest.mark.skip
def test_A3():
    a = clique_matrix(3)

    solve_all( a, np.array([6,3]))
    solve_all( a, np.array([1,1]))
    solve_all( a, np.array([3,-3]))

#@pytest.mark.skip
def test_A4():
    a = clique_matrix( 4)

    solve_all( a, np.array([1,1,1]))
    solve_all( a, np.array([8,4,0]))

def test_A5():
    a = clique_matrix( 5)

    solve_all( a, np.array([1,1,1,1]))
    solve_all( a, np.array([10,5,0,0]))

def test_A10():
    a = clique_matrix( 10)

    solve_all( a, np.array([1,1,1,1,1,1,1,1,1]))
    solve_all( a, np.array([20,10,0,0,0,0,0,0,0]))
    solve_all( a, np.array([40,20,10,0,0,0,0,0,0]))
    solve_all( a, np.array([80,40,20,10,0,0,0,0,0]))
