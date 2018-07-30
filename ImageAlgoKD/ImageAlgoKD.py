from pylab import *
import pandas as pd
from DataStructure import *
from timeit import default_timer as timer

cudaIsAvailable = True
openclIsAvailable = True

try:
    import pycuda
except ModuleNotFoundError:
    cudaIsAvailable = False

try:
    import pyopencl
except ModuleNotFoundError:
    openclIsAvailable = False

if cudaIsAvailable:
    from ImageAlgoKD_kernel_cuda import *

if openclIsAvailable:
    from ImageAlgoKD_kernel_opencl import *
    from Utilities import openclInfo

class ImageAlgoKD():
    def __init__(self,
                 MAXDISTANCE        = 10,
                 KERNEL_R           = 1.0,
                 KERNEL_R_NORM      = 1.0,
                 KERNEL_R_POWER     = 0.0,
                 DECISION_RHO_KAPPA = 4.0,
                 DECISION_NHD       = 1.0,
                 CONTINUITY_NHD     = 1.0
                ):
        self.MAXDISTANCE    = np.float32(MAXDISTANCE)
        self.KERNEL_R       = np.float32(KERNEL_R)
        self.KERNEL_R_NORM  = np.float32(KERNEL_R_NORM)
        self.KERNEL_R_POWER = np.float32(KERNEL_R_POWER)
        self.DECISION_RHO_KAPPA = np.float32(DECISION_RHO_KAPPA)
        self.DECISION_NHD   = np.float32(DECISION_NHD)
        self.CONTINUITY_NHD = np.float32(CONTINUITY_NHD)

    def setInputsPoints(self, points):
        self.points = points

    def run(self, method="numpy", deviceID=0, blockSize=1):
        self.points.reset()
        start = timer()
        # get decision variables
        if   (method == "cuda")   & cudaIsAvailable:
            self.getDecisionVariables_cuda(blockSize)
            self.method = "cuda"
        
        elif (method == "cudabin") & cudaIsAvailable:
            self.getDecisionVariables_cudabin(blockSize)
            self.method = "cudabin"

        elif (method == "opencl") & openclIsAvailable:
            self.getDecisionVariables_opencl(deviceID,blockSize)
            self.method = "opencl"
        
        elif (method == "openclbin") & openclIsAvailable:
            self.getDecisionVariables_openclbin(deviceID,blockSize)
            self.method = "openclbin"
        
        elif (method == "numpybin"): #& (self.points.k in [2,3,4]):
            self.getDecisionVariables_numpybin()
            self.method = "numpybin"

        else:
            self.getDecisionVariables_numpy()
            self.method = "numpy"

        print("clustering finished!")
        end = timer()

        # make clusters
        self.getClusteringResults()

        # print timing
        self.runtime = end-start
        print("Run time with {} is {:7.4f} ms, in which rho time is {:7.4f} ms".format(self.method, self.runtime*1000, self.rhotime*1000  ))
        
    

    def getClusteringResults(self):

        n, k = self.points.n, self.points.k

        self.DECISION_RHO = self.points.rho.max()/self.DECISION_RHO_KAPPA

        
        clusterID  = -np.ones(n,int)
        # convert rhorank to argsortrho, O(N)
        argsortrho = np.empty(n,int)
        argsortrho[self.points.rhorank] = np.arange(n)

        # find seeds and their seedIDs
        isSeed  = (self.points.rho>self.DECISION_RHO) & (self.points.nhd>self.DECISION_NHD)
        rhoSeed = self.points.rho[isSeed]
        argsortrhoSeed = rhoSeed.argsort(kind='mergesort')[::-1]
        seedID = np.empty(rhoSeed.size, int)
        seedID[argsortrhoSeed] = np.arange(rhoSeed.size)

        # clusterIDs of seeds are their seedIDs
        clusterID[isSeed] = seedID

        # asign points to seeds from the highest rho to lowest rho, O(N)
        for ith in range(n):
            i = argsortrho[ith]
            # if not have a clusterID and satisfify continuity
            if  (clusterID[i]<0) & (self.points.nhd[i]<self.CONTINUITY_NHD):
                clusterID[i] = clusterID[self.points.nh[i]]
        
        # finally save isSeed and clusterID to points
        self.points.isSeed    = isSeed
        self.points.clusterID = clusterID

    def getDecisionVariables_cuda(self, blockSize=1024):

        n, k = self.points.n, self.points.k
        
        # allocate memery on device for points and decision parameters
        d_cords   = cuda.mem_alloc(self.points.cords.nbytes)
        d_wegiths = cuda.mem_alloc(self.points.weights.nbytes)
        d_rho     = cuda.mem_alloc(self.points.rho.nbytes)
        d_rhorank = cuda.mem_alloc(self.points.rhorank.nbytes)
        d_nh      = cuda.mem_alloc(self.points.nh.nbytes)
        d_nhd     = cuda.mem_alloc(self.points.nhd.nbytes)

        # copy memergy to device
        cuda.memcpy_htod( d_cords  , self.points.cords )
        cuda.memcpy_htod( d_wegiths, self.points.weights )

        # run KERNEL on device
        start = timer()
        rho_cuda(   d_rho, d_cords, d_wegiths,
                    # algorithm parameters
                    n, k, self.KERNEL_R, self.KERNEL_R_NORM, self.KERNEL_R_POWER,
                    grid  = (int(n/blockSize)+1,1,1),
                    block = (int(blockSize),1,1) )

        cuda.memcpy_dtoh(self.points.rho,d_rho)
        end = timer()
        self.rhotime = end-start

        rhoranknh_cuda( d_rhorank, d_nh, d_nhd, d_cords, d_rho,
                        # algorithm paramters
                        n, k, self.MAXDISTANCE,
                        grid  = (int(n/blockSize)+1,1,1),
                        block = (int(blockSize),1,1) )

        # copy memery from device to host
        
        cuda.memcpy_dtoh(self.points.rhorank,d_rhorank)
        cuda.memcpy_dtoh(self.points.nh,d_nh)
        cuda.memcpy_dtoh(self.points.nhd,d_nhd)
        
        # release globle memery on device
        d_cords.free()
        d_wegiths.free()
        d_rho.free()
        d_rhorank.free()
        d_nh.free()
        d_nhd.free()


    def getDecisionVariables_opencl(self, deviceID=0, blockSize=1):

        context,prg = openclKernel(DeviceID=deviceID)
        queue = cl.CommandQueue(context)

        n, k = self.points.n, self.points.k

        LOCALSIZE  = int(blockSize)
        GLOBALSIZE = (int(n/LOCALSIZE)+1)*LOCALSIZE

        # allocate memery on device for points and decision parameters
        d_rho     = cl.Buffer(context, cl.mem_flags.READ_WRITE, self.points.rho.nbytes)
        d_rhorank = cl.Buffer(context, cl.mem_flags.READ_WRITE, self.points.rhorank.nbytes)
        d_nh      = cl.Buffer(context, cl.mem_flags.READ_WRITE, self.points.nh.nbytes)
        d_nhd     = cl.Buffer(context, cl.mem_flags.READ_WRITE, self.points.nhd.nbytes)

        # copy memergy to device
        d_cords   = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.points.cords)
        d_wegiths = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.points.weights)

        start = timer()
        # run KERNEL on device
        prg.rho_opencl( queue, (GLOBALSIZE,), (LOCALSIZE,),
                        d_rho, d_cords, d_wegiths,
                        # algorithm parameters
                        n, k, self.KERNEL_R, self.KERNEL_R_NORM, self.KERNEL_R_POWER
                        )
        cl.enqueue_copy(queue, self.points.rho, d_rho)
        end = timer()
        self.rhotime = end-start

        prg.rhoranknh_opencl(   queue, (GLOBALSIZE,), (LOCALSIZE,),
                                d_rhorank, d_nh, d_nhd, d_cords, d_rho,
                                # algorithm paramters
                                n, k, self.MAXDISTANCE
                                )

        # copy memery from device to host
        cl.enqueue_copy(queue, self.points.rhorank, d_rhorank)
        cl.enqueue_copy(queue, self.points.nh, d_nh)
        cl.enqueue_copy(queue, self.points.nhd, d_nhd)

        # release globle memery on device
        d_cords.release()
        d_wegiths.release()
        d_rho.release()
        d_rhorank.release()
        d_nh.release()
        d_nhd.release()

    def getDecisionVariables_numpy(self):

        n, k = self.points.n, self.points.k
        ###########################
        # find rho 
        ###########################
        start = timer()
        rho = []
        for i in range(n):
            dr = self._norm2Distance(self.points.cords, self.points.cords[i])
            local = (dr<self.KERNEL_R)
            if self.KERNEL_R_POWER == 0:
                irho = np.sum( self.points.weights[local] )
            else:
                irho = np.sum( self.points.weights[local] * np.exp( - (dr[local]/self.KERNEL_R_NORM)**self.KERNEL_R_POWER ))
    
            rho.append(irho)
        rho = np.array(rho)

        end = timer()
        self.rhotime = end-start

        ###########################
        # find rhorank
        ###########################
        argsortrho = rho.argsort(kind='mergesort')[::-1]
        rhorank = np.empty(rho.size, int)
        rhorank[argsortrho] = np.arange(rho.size)

        ###########################
        # find NearstHiger and distance to NearestHigher
        ###########################
        nh,nhd = [],[]
        for i in range(n):
            irank = rhorank[i]
            
            higher = rhorank<irank
            # if no points is higher
            if not (True in higher): 
                nh. append(i)
                nhd.append(self.MAXDISTANCE)
            else:
                drr  = self._norm2Distance(self.points.cords[higher], self.points.cords[i])
                temp = np.arange(rho.size)[higher]
                nh. append(temp[np.argmin(drr)])
                nhd.append(np.min(drr))
                    
        nh = np.array(nh)
        nhd= np.array(nhd)

        # save the decision variables to points
        self.points.rho = rho
        self.points.rhorank = rhorank
        self.points.nh  = nh
        self.points.nhd = nhd


    def getDecisionVariables_cudabin(self, blockSize=1024):

        n, k = self.points.n, self.points.k
        
        # allocate memery on device for points and decision parameters
        d_cords   = cuda.mem_alloc(self.points.cords.nbytes)
        d_wegiths = cuda.mem_alloc(self.points.weights.nbytes)

        d_nnbinHead = cuda.mem_alloc(self.points.point_idxNNBinsHead.nbytes)
        d_nnbinSize = cuda.mem_alloc(self.points.point_idxNNBinsSize.nbytes)
        d_nnbinList = cuda.mem_alloc(self.points.idxNNBinsList.nbytes)
        d_idxPointsHead = cuda.mem_alloc(self.points.bin_idxPointsHead.nbytes)
        d_idxPointsSize = cuda.mem_alloc(self.points.bin_idxPointsSize.nbytes)
        d_idxPonitsList = cuda.mem_alloc(self.points.idxPonitsList.nbytes)


        d_rho     = cuda.mem_alloc(self.points.rho.nbytes)
        d_rhorank = cuda.mem_alloc(self.points.rhorank.nbytes)
        d_nh      = cuda.mem_alloc(self.points.nh.nbytes)
        d_nhd     = cuda.mem_alloc(self.points.nhd.nbytes)

        # copy memergy to device
        cuda.memcpy_htod( d_cords  , self.points.cords )
        cuda.memcpy_htod( d_wegiths, self.points.weights )

        cuda.memcpy_htod( d_nnbinHead, self.points.point_idxNNBinsHead )
        cuda.memcpy_htod( d_nnbinSize, self.points.point_idxNNBinsSize )
        cuda.memcpy_htod( d_nnbinList, self.points.idxNNBinsList )
        cuda.memcpy_htod( d_idxPointsHead, self.points.bin_idxPointsHead )
        cuda.memcpy_htod( d_idxPointsSize, self.points.bin_idxPointsSize )
        cuda.memcpy_htod( d_idxPonitsList, self.points.idxPonitsList )

        # run KERNEL on device
        start = timer()
        rho_cudabin(d_rho, d_cords, d_wegiths,
                    d_nnbinHead,
                    d_nnbinSize,
                    d_nnbinList,
                    d_idxPointsHead, 
                    d_idxPointsSize,
                    d_idxPonitsList,
                    # algorithm parameters
                    n, k, self.KERNEL_R, self.KERNEL_R_NORM, self.KERNEL_R_POWER,
                    grid  = (int(n/blockSize)+1,1,1),
                    block = (int(blockSize),1,1) )
        
        cuda.memcpy_dtoh(self.points.rho,d_rho)
        end = timer()
        self.rhotime = end-start

        rhoranknh_cuda( d_rhorank, d_nh, d_nhd, d_cords, d_rho,
                        # algorithm paramters
                        n, k, self.MAXDISTANCE,
                        grid  = (int(n/blockSize)+1,1,1),
                        block = (int(blockSize),1,1) )

        # copy memery from device to host
        
        cuda.memcpy_dtoh(self.points.rhorank,d_rhorank)
        cuda.memcpy_dtoh(self.points.nh,d_nh)
        cuda.memcpy_dtoh(self.points.nhd,d_nhd)
        
        # release globle memery on device
        d_cords.free()
        d_wegiths.free()
        d_rho.free()
        d_rhorank.free()
        d_nh.free()
        d_nhd.free()


    def getDecisionVariables_openclbin(self, deviceID=0, blockSize=1):

        context,prg = openclKernel(DeviceID=deviceID)
        queue = cl.CommandQueue(context)

        n, k = self.points.n, self.points.k

        LOCALSIZE  = int(blockSize)
        GLOBALSIZE = (int(n/LOCALSIZE)+1)*LOCALSIZE

        # allocate memery on device for points and decision parameters
        d_rho     = cl.Buffer(context, cl.mem_flags.READ_WRITE, self.points.rho.nbytes)
        d_rhorank = cl.Buffer(context, cl.mem_flags.READ_WRITE, self.points.rhorank.nbytes)
        d_nh      = cl.Buffer(context, cl.mem_flags.READ_WRITE, self.points.nh.nbytes)
        d_nhd     = cl.Buffer(context, cl.mem_flags.READ_WRITE, self.points.nhd.nbytes)

        # copy memergy to device
        d_cords   = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.points.cords)
        d_wegiths = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.points.weights)
        
        d_nnbinHead = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.points.point_idxNNBinsHead)
        d_nnbinSize = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.points.point_idxNNBinsSize)
        d_nnbinList = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.points.idxNNBinsList)
        d_idxPointsHead = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.points.bin_idxPointsHead)
        d_idxPointsSize = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.points.bin_idxPointsSize)
        d_idxPonitsList = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.points.idxPonitsList)


        # run KERNEL on device
        start = timer()
        prg.rho_openclbin( queue, (GLOBALSIZE,), (LOCALSIZE,),
                        d_rho, d_cords, d_wegiths, 
                        d_nnbinHead,
                        d_nnbinSize,
                        d_nnbinList,
                        d_idxPointsHead, 
                        d_idxPointsSize,
                        d_idxPonitsList,
                        # algorithm parameters
                        n, k, self.KERNEL_R, self.KERNEL_R_NORM, self.KERNEL_R_POWER
                        )
        cl.enqueue_copy(queue, self.points.rho, d_rho)
        
        end = timer()
        self.rhotime = end-start

        
        prg.rhoranknh_opencl(   queue, (GLOBALSIZE,), (LOCALSIZE,),
                                d_rhorank, d_nh, d_nhd, d_cords, d_rho,
                                # algorithm paramters
                                n, k, self.MAXDISTANCE
                                )

        # copy memery from device to host
        cl.enqueue_copy(queue, self.points.rhorank, d_rhorank)
        cl.enqueue_copy(queue, self.points.nh, d_nh)
        cl.enqueue_copy(queue, self.points.nhd, d_nhd)

        # release globle memery on device
        d_cords.release()
        d_wegiths.release()

        d_nnbinHead.release()
        d_nnbinSize.release()
        d_nnbinList.release()
        d_idxPointsHead.release()
        d_idxPointsSize.release()
        d_idxPonitsList.release()

        d_rho.release()
        d_rhorank.release()
        d_nh.release()
        d_nhd.release()


    def getDecisionVariables_numpybin(self):
        '''
        numpy for bined data
        '''

        n, k = self.points.n, self.points.k

        ###########################
        # find rho, O(n)
        ###########################
        start = timer()
        # get rho by looping over bins
        rho = np.zeros(n)
        # loop over location keys
        for key in self.points.keys:
            idxPoints,keyNeighbors,_ = self.points.bins[key]
            idxNeighbors = self.points.getIdxNeighborsFromKey(key)

            # loop over points in the location key
            for idxPoint in idxPoints:
                icord   = np.take(self.points.cords,idxPoint,axis=0)
                jcord   = np.take(self.points.cords,idxNeighbors,axis=0)
                jweight = np.take(self.points.weights,idxNeighbors,axis=0)
                # get density of the points and save to rho
                dr = self._norm2Distance(icord,jcord)

                local = dr<self.KERNEL_R
                if self.KERNEL_R_POWER == 0:
                    irho = np.sum( jweight[local] )
                else:
                    irho = np.sum(  weight[local] * np.exp( - (dr[local]/self.KERNEL_R_NORM)**self.KERNEL_R_POWER ))

                rho[idxPoint] = irho
        


        # get rho by looping over points
        # rho = []
        # for i in range(points.n):
        #     j = points.getIdxNeighborsFromIdx(i)

        #     icord = points.cords[i]
        #     jcord = points.cords[j]

        #     jweight = points.weights[j]

        #     dr = np.sqrt( np.sum((jcord-icord)**2, axis=-1) )
        #     local = dr<KERNEL_R
        #     irho = np.sum( jweight[local] )
        #     rho.append(irho)
        # rho = np.array(rho)
        end = timer()
        self.rhotime = end-start
        ###########################
        # find rhorank, O(nlogn) based on mergesort
        ###########################
        argsortrho = rho.argsort(kind='mergesort')[::-1]
        rhorank = np.empty(rho.size, int)
        rhorank[argsortrho] = np.arange(rho.size)

        ###########################
        # find NearstHiger and distance to NearestHigher,O(mn)
        ###########################
        nh = np.arange(n)
        nhd = np.zeros(n)
        
        self.nNNSearch, self.nGlobalSearch = 0,0

        # loop over keys
        for key in self.points.keys:
            idxPoints,stencilKeys,_ = self.points.bins[key]

            # query neighers
            idxNeighbors = self.points.getIdxNeighborsFromKey(key)
            rhorankNeighbors = rhorank[idxNeighbors]
            cordNeighbors = self.points.cords[idxNeighbors]
            nNeighbors = len(idxNeighbors)

            # nNeighors exit
            # if nNeighbors > 1:

            # loop over points in the key
            for idxPoint in idxPoints:
                irhorank = rhorank[idxPoint]
                icord = self.points.cords[idxPoint]
                
                #isearchGlobal = True
                # search NH in neighbors
                higherInNeighbors = rhorankNeighbors < irhorank
                if (True in higherInNeighbors):
                    self.nNNSearch += 1
                    #isearchGlobal = False


                    dr = self._norm2Distance( icord, cordNeighbors[higherInNeighbors])
                    idxHighers = np.arange(nNeighbors)[higherInNeighbors]

                    inhd = np.min(dr)
                    
                    # requiring inhd in save area
                    if inhd<self.KERNEL_R:
                        #inh  = idxNeighbors[ idxHighers[np.argmin(dr)] ]
                        inh = np.array(idxNeighbors)[ idxHighers[ np.argwhere(dr==inhd).reshape(-1) ] ].min()

                        nh [idxPoint] = inh
                        nhd[idxPoint] = inhd
                    else:
                        nh[idxPoint],nhd[idxPoint] = self._searchNHGlobal(idxPoint,rhorank)
                
                 # search NH in globe
                else:
                    nh[idxPoint],nhd[idxPoint] = self._searchNHGlobal(idxPoint,rhorank)



        # save the decision variables to points
        self.points.rho = rho
        self.points.rhorank = rhorank
        self.points.nh  = nh
        self.points.nhd = nhd

        self.effNN = self.nNNSearch/(self.nGlobalSearch+self.nNNSearch)


    ## private functions
    def _norm2Distance(self, p1,p2):
        return np.sqrt( np.sum((p1-p2)**2, axis=-1) )

    def _searchNHGlobal(self,idxPoint,rhorank):
        
        self.nGlobalSearch += 1
        higher = rhorank < rhorank[idxPoint]
        # find higher
        if (True in higher): 
            dr  = self._norm2Distance(self.points.cords[idxPoint],self.points.cords[higher])
            idxHighers = np.arange(self.points.n)[higher]
            inh  = idxHighers[np.argmin(dr)]
            inhd = np.min(dr)
        # no higher are found
        else:
            inh = idxPoint
            inhd = self.MAXDISTANCE

        return inh,inhd