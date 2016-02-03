import os
import sys
import numpy as np
import math
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../solvers')
import linAlgSolveCy
import linAlgSolve

relTol = 1.0e-3
absTol = 0.0

# A x = B
A1 = np.array([[1.0,0.0],
               [0.0,1.0]])
B1 = np.array([1.0,0.0])
x1 = np.array([1.0,0.0])

A2 = np.array([[1.0,0.0,0.0],
               [0.0,1.0,0.0],
               [0.0,0.0,1.0]])
B2 = np.array([1.0,0.0,0.0])
x2 = np.array([1.0,0.0,0.0])

def testSolvers():
  def test(array1,array2):
    assert np.allclose(array1,array2,relTol,absTol)

  def testCase(A,x,B):
    x_Jacobi             = linAlgSolve.jacobi(A,B,relTol,absTol)
    x_Jacobi_Cython      = linAlgSolveCy.jacobi(A,B,relTol,absTol)
    x_GaussSeidel        = linAlgSolve.gaussSeidel(A,B,relTol,absTol)
    x_GaussSeidel_Cython = linAlgSolveCy.gaussSeidel(A,B,relTol,absTol)
    x_Direct             = linAlgSolve.direct(A,B)
    x_Direct_Cython      = linAlgSolveCy.direct(A,B)
 
    test(x_Jacobi,x)
    test(x_Jacobi_Cython,x)
    test(x_GaussSeidel,x)
    test(x_GaussSeidel_Cython,x)
    test(x_Direct,x)
    test(x_Direct_Cython,x)

  testCase(A1,x1,B1)
  testCase(A2,x2,B2)
