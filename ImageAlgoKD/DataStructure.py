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



class Points_KDBin(Points):
    def __init__(self, cords, KERNEL_R):

        super().__init__(cords)
        self.binSize = KERNEL_R
        self.makeStencil()
        self.makeKDBins()
        self.makeIdxPointList()
        self.makeNNBinList()

    
    def makeKDBins(self):

        # push references of points to the hashmap
        bins = {}
        for i in range(self.n):

            key = self.getKeyFromCord( self.cords[i], self.binSize) 
            if key in bins:
                bins[key] = bins[key]+[i]
            else:
                bins[key] = [i]

        self.keys = bins.keys()

        # push keys of stencils to hashmap
        for idx,key in enumerate(self.keys):
            key = np.array(key)

            stencilKeys = []
            for i in self.stencil:
                ikey = key + i
                ikey = tuple(ikey)
                if ikey in bins:
                    stencilKeys += [ikey]
            key = tuple(key)
            bins[key] = ( bins[key], stencilKeys, idx )

        self.bins = bins

    def makeStencil(self):
        if self.k == 2:
            vx,vy = np.mgrid[-1.0:2.0,-1.0:2.0]
        elif self.k == 3:
            vx,vy = np.mgrid[-1.0:2.0,-1.0:2.0,-1.0:2.0]
        elif self.k == 4:
            vx,vy = np.mgrid[-1.0:2.0,-1.0:2.0,-1.0:2.0,-1.0:2.0]
        else:
            print(' Dimention not supported because inefficient using kdbin with large k')

        vx,vy = vx.reshape(-1), vy.reshape(-1)
        stencil = np.c_[vx,vy] 
        stencil *= self.binSize
        self.stencil = stencil
        

    def makeIdxPointList(self):
        idxPonitsList = []

        bin_idxPointsHead = []
        bin_idxPointsSize = []

        ihead = 0
        for v in self.bins.values():
            bin_idxPointsHead.append(ihead)
    
            idxPonits = v[0]
    
            idxPonitsList += idxPonits
    
            size = len(idxPonits)
            bin_idxPointsSize.append(size)
            ihead += size
        
        self.idxPonitsList = np.array(idxPonitsList).astype(np.int32)
        self.bin_idxPointsHead = np.array(bin_idxPointsHead).astype(np.int32)
        self.bin_idxPointsSize = np.array(bin_idxPointsSize).astype(np.int32)

    def makeNNBinList(self):
        keyNNBinsList = []
        idxNNBinsList = []
        
        bin_idxNNBinsHead = []
        bin_idxNNBinsSize = []

        ihead = 0
        for v in self.bins.values():
            bin_idxNNBinsHead.append(ihead)
    
            keyNNBins = v[1]
            keyNNBinsList += keyNNBins
    
            size = len(keyNNBins)
            bin_idxNNBinsSize.append(size)
            ihead += size

        for key in keyNNBinsList:
            idxNNBinsList.append(self.bins[key][2])
        
        self.idxNNBinsList = np.array(idxNNBinsList).astype(np.int32)
        self.bin_idxNNBinsHead = np.array(bin_idxNNBinsHead).astype(np.int32)
        self.bin_idxNNBinsSize = np.array(bin_idxNNBinsSize).astype(np.int32)

        ## give bin nnBinsHead and size to points
        self.point_idxNNBinsHead = np.empty(self.n).astype(np.int32)
        self.point_idxNNBinsSize = np.empty(self.n).astype(np.int32)
        for i, v in enumerate(self.bins.values()):
            idxPonits = v[0]
            for idx in idxPonits:
                self.point_idxNNBinsHead[idx] = self.bin_idxNNBinsHead[i]
                self.point_idxNNBinsSize[idx] = self.bin_idxNNBinsSize[i]

    ## public functionsß

    def getKeyFromCord(self,cord,binSize):
        key = np.floor(cord/ binSize)
        key *= binSize
        key = tuple(key)
        return key


    def getIdxNeighborsFromKey(self, key):
        idxPoints,keyNeighbors,_ = self.bins[key]
        idxNeighbors = []
        for key in keyNeighbors:
            idxNeighbors += self.bins[key][0]
        return idxNeighbors

    def getIdxNeighborsFromIdx(self, idxPoint):

        a = self.point_idxNNBinsHead[idxPoint]
        b = a + self.point_idxNNBinsSize[idxPoint]
        idxNNBins = self.idxNNBinsList[a:b]

        idxNNPoints = np.array([]).astype(np.int32)
        for idxNNBin in idxNNBins:
            a = self.bin_idxPointsHead[idxNNBin]
            b = a + self.bin_idxPointsSize[idxNNBin]
            idxNNPoints = np.r_[idxNNPoints, self.idxPonitsList[a:b] ]
        
        return idxNNPoints

