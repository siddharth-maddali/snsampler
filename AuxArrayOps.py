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

def multiskew( veclist ):
    """
    multiskew( veclist ):
    
    Returns a list of skew matrices corresponding to each 3D column 
    vector in the 3xN-size input `veclist`.

    Inputs: 
    `veclist`: 2D array of size 3xN representing a list of N vectors

    Output: 
    A 3D array of size 3x3xN, representing the N skew-symmetric 
    cross-product matrices corresponding to each of the 3D vectors 
    in the input.
    """
    skew0 = np.array( 
        [ 
            [ 0., 0., 0. ], 
            [ 0., 0., -1. ], 
            [ 0., 1., 0. ]
        ]
    ).reshape( 3, 3, 1 ).repeat( veclist.shape[-1], axis=2 )
    skew1 = np.array( 
        [ 
            [ 0., 0., 1. ], 
            [ 0., 0., 0. ], 
            [ -1., 0., 0. ]
        ]
    ).reshape( 3, 3, 1 ).repeat( veclist.shape[-1], axis=2 )
    skew2 = np.array( 
        [ 
            [ 0., -1., 0. ], 
            [ 1., 0., 0. ], 
            [ 0., 0., 0. ]
        ]
    ).reshape( 3, 3, 1 ).repeat( veclist.shape[-1], axis=2 )

    skewlist = [ skew0, skew1, skew2 ]
    result = ftools.reduce( 
        np.add, 
        [ 
            veclist[n,:].reshape( 1, 1, -1 ) * skewlist[n] 
            for n in [ 0, 1, 2 ] 
        ] 
    )
    return result


