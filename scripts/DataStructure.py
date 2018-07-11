from pylab import *

class Points():
    def __init__(self, cords):

        self.cords = cords.astype(np.float32)
        self.n = np.int32( cords.shape[0] )
        self.k = np.int32( cords.shape[1] )

        self.weights = np.ones (self.n).astype(np.float32) 
        self.reset() 

    def reset(self):
        # decision variables
        self.rho     = np.zeros(self.n).astype(np.float32)
        self.rhorank = np.zeros(self.n).astype(np.int32)
        self.nh      = np.zeros(self.n).astype(np.int32)
        self.nhd     = np.zeros(self.n).astype(np.float32)

        # clustering results
        self.isSeed    = None
        self.clusterID = None

    def setWeights(self, weights):
        self.weights = weights.astype(np.float32)



        
        
