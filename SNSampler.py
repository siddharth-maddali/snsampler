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



