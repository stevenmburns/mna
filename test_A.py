import numpy as np
import numpy.linalg as la
import pytest
import cvxpy as cvx


@pytest.mark.skip
def test_A3():
    a = np.array( [[-1,0,1],[0,-1,-1]])
    b = np.block( [[np.zeros((2,2)),a],[a.T, np.identity(3)*-.001]])

    source = np.array([6,3,0,0,0])
    print(la.inv(b).dot(source))

    source = np.array([1,1,0,0,0])
    print(la.inv(b).dot(source))

    source = np.array([3,-3,0,0,0])
    print(la.inv(b).dot(source))

def test_A4():
    a = np.array( [[-1,0,0,1,1,0],
                   [0,-1,0,-1,0,1],
                   [0,0,-1,0,-1,-1]])
    b = np.block( [[np.zeros((3,3)),a],[a.T, np.identity(6)*-.001]])

    gram = a.dot(a.T)

    print()
    source = np.array([1,1,1,0,0,0,0,0,0])
    print( f'source {source}')
    print( f'resistors {la.inv(b).dot(source)[3:]}')
    print( f'l2opt {a.T.dot(la.inv(gram).dot( source[:3]))}')

    x = cvx.Variable(6)
    objective = cvx.Minimize(cvx.norm(x,1))
    constraints = [a * x == source[:3]]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)

    print( f'l1opt {np.array(x.value)}')

    source = np.array([8,4,0,0,0,0,0,0,0])
    print( f'source {source}')
    print( f'resistors {la.inv(b).dot(source)[3:]}')
    print( f'l2opt {a.T.dot(la.inv(gram).dot( source[:3]))}')

    x = cvx.Variable(6)
    objective = cvx.Minimize(cvx.norm(x,1))
    constraints = [a * x == source[:3]]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)

    print( f'l1opt {np.array(x.value)}')
