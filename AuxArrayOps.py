#####################################################################
#
#    AuxArrayOps:
#        Contains functions to perform vectorized tensor-wise operations. 
#        Requires the list of tensors to be cast as a 3D array with the 
#        third dimension spanning the number of tensors in the list.
#
#    Siddharth Maddali
#    Argonne National Laboratory 
#    smaddali@alumni.cmu.edu
#    February 2020
#
#####################################################################

import numpy as np
import functools as ftools

def multimatmul( a, b ):
    """
    multimatmul( a, b ):

    Returns the elementwise matrix product of matrix lists `a` and `b`. 
    This is completely vectorized code that uses numpy's in-built 
    matmul and rollaxis functions.

    Inputs:
    `a`: 3D array of size ( D1, M, N ) representing a list of N 
    matrices, each of size D1xM.
    `b`: 3D array of size ( M, D2, N ) representing a list of N 
    matrices, each of size MxD2

    Outputs: 
    A 3D array of size (D1, D2, N ) representing a list of N 
    matrices of size D1xD2.

    """
    return np.rollaxis(
        np.matmul( 
            np.rollaxis( a, 2 ), 
            np.rollaxis( b, 2 )
        ), 
        0, 3
    )

