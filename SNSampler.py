#####################################################################
#
#    SNSampler: 
#        Class to return a set of points that sample the 
#        unit sphere in N dimensions (S^N-1) nearly uniformly.
#        Reference: 
#        A. Yershova and S. M. LaValle.
#        Deterministic sampling methods for spheres and SO(3).
#        In Proceedings IEEE International Conference 
#        on Robotics and Automation, 2004.
#
#    Siddharth Maddali
#    Argonne National Laboratory
#    smaddali@alumni.cmu.edu
#    February 2020
#
#####################################################################

import numpy as np
from scipy.spatial.transform import Rotation

import snsampler.AuxArrayOps as aux  # enables some useful array operations

class SNSampler: 

    def __init__( self, n_dims=3, n_samples=5 ):

        grids = np.meshgrid( 
            *[ np.linspace( -1., 1., n_samples ) for n in list( range( n_dims-1 ) ) ] 
        ) # this returns a tuple of grids (eg. for n_dims=3, this returns x, y, z grid arrays)

        ptsNm1d = np.concatenate( 
            tuple( [ this.reshape( 1, -1 ) for this in grids ] ), 
            axis=0
        ) # gives low-dimensional representation of points along only one pair of opposing faces of the high-dimensional cube

        ptsNd = np.concatenate( 
            tuple( 
                [ 
                    np.concatenate( 
                        ( ptsNm1d, scl*np.ones( ( 1, ptsNm1d.shape[-1] ) ) ), 
                        axis=0 
                    ) for scl in [ -1., 1. ] 
                ] 
            ), 
            axis=1
        ) # embeds the above lower-dimensional points onto the actual opposite faces of the high-dimensional cube
        pts = np.unique( 
            np.concatenate( 
                tuple( [ np.roll( ptsNd, n, axis=0 ) for n in list( range( n_dims ) ) ] ), 
                axis=1 
            ), 
            axis=1
        ) # augments the above array to obtain the sample points along the other faces of the cube
        
        normlizr = np.sqrt( ( pts**2 ).sum( axis=0 ) ).reshape( 1, -1 ).repeat( n_dims, axis=0 )
        self.sample_points = pts / normlizr
            # normalizes each column of the above array to get unit-norm vectors
        return

    def removeRedundantRotations( self ):
        """
        Because unit quaternions q and -q denote the same physical rotation, 
        this method only keeps the 'canonical' ones and removes the redundant 
        ones computed in the constructor __init__.
        NOTE: only makes sense to do this if n_dims = 4.
        """
        self.sample_points = self.sample_points[ :, np.where( self.sample_points[0,:] > 0. )[0] ]
        return


    def getRMatsFromQuats( self ):
        """
        getRMatsFromQuats():
        Computes the rotation matrices corresponding to the set of unit quaternions.
        Uses Built-in methods of the Rotation class for rapid conversion.

        Input: 
        None

        Output: 
        A 3x3xN array with each 3x3 slice denoting the orthogonal rotation matrix computed from 
        the unit quaternion that is the corresponding column of self.sample_points.
        """
        R = Rotation.from_quat( self.sample_points.T )
        self.RMat           = np.rollaxis( R.as_matrix(), 0, 3 )
        self.RMatInverse    = np.rollaxis( R.inv().as_matrix(), 0, 3 )
        return

    def getTransformationSequence( self, start=0, stride=1 ):
        """
        getTansformationSequence():
        Computes the **differential** rotational operators (in matrix form) that 
        transform each rotation self.RMat[:,:,start+n*stride] into self.RMat[:,:,start+(n+1)*stride]. In other 
        words, computes the 3x3xN array of matrices ARR such that:

            ARR[:,:,start+n*stride] @ self.RMat[:,:,start+n*stride] = self.RMat[:,:,start+(n+1)*stride].

        Here N is equal to len( list( range( start, self.RMat.shape[-1], stride ) ).
        This is something like the 'diff' function in Matlab, but for rotations. 
        It can be used to transform extremely large geometric datasets in a 
        sequential manner, without having to undo the previous transformation.

        Input: 
        None

        Output: 
        A 3x3xN array (same size as self.RMat).

        The first elements of ARR and self.RMat are the same (i.e., the difference 
        is assumed starting from the identity rotation).
        
        """
        try:
            these = list( range( start, self.RMat.shape[-1], stride ) )
            myarr = np.zeros( ( 3, 3, len( these ) ) )
            myarr[:,:,0] = self.RMat[:,:,these[0]]
            myarr[:,:,1:] = aux.multimatmul( 
                self.RMat[:,:,these[1:]], 
                self.RMatInverse[:,:,these[:-1]] 
            )
            return myarr, these
        except AttributeError:  # i.e., if self.RMat and/or self.RMatInverse not found ...
            print( 'ERROR: Try running getRMatsFromQuats() first. ' )
            return

