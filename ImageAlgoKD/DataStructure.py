from pylab import *

class Points():
    def __init__(self, cords):
        self.inbins = False

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

    def binPoints(self,KERNEL_R):
        if self.k in [2,3,4]:

            # push references of points to the hashmap
            bins = {}
            for i in range(self.n):
                key = np.floor(self.cords[i]/KERNEL_R)
                key *= KERNEL_R
                key = tuple(key)
                if key in bins:
                    bins[key] = bins[key]+[i]
                else:
                    bins[key] = [i]

            self.keys = bins.keys()

            # getting stencil
            if self.k == 2:
                vx,vy = np.mgrid[-1.0:2.0,-1.0:2.0]
            elif self.k == 3:
                vx,vy = np.mgrid[-1.0:2.0,-1.0:2.0,-1.0:2.0]
            elif self.k == 4:
                vx,vy = np.mgrid[-1.0:2.0,-1.0:2.0,-1.0:2.0,-1.0:2.0]

            vx,vy = vx.reshape(-1),vy.reshape(-1)
            stencil = np.c_[vx,vy] 
            stencil *= KERNEL_R
            self.stencil = stencil


            # push keys of stencils to hashmap
            for key in self.keys:
                key = np.array(key)
                # collect index in neighering bins
                stencilKeys = []
                for i in self.stencil:
                    ikey = key + i
                    ikey = tuple(ikey)
                    if ikey in bins:
                        stencilKeys += [ikey]
                key = tuple(key)
                bins[key] = ( bins[key], stencilKeys )

            self.bins = bins
            self.isbinned = True

        else:
            print(" Bin not available")


    def getIdxNeighbors(self, key):
        if self.isbinned:

            idxPoints,keyNeighbors = self.bins[key]

            idxNeighbors = []

            for key in keyNeighbors:
                idxNeighbors += self.bins[key][0]

            return idxNeighbors
        else:
            print(' Bin not available')

