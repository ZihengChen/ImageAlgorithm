from pylab import *
import pandas as pd
from DataStructure import *

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
    from utilities import openclInfo

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
        # get decision variables
        if   (method == "cuda")   & cudaIsAvailable:
            self.getDecisionVariables_cuda(blockSize)
        elif (method == "opencl") & openclIsAvailable:
            self.getDecisionVariables_opencl(deviceID,blockSize)
        else:
            self.getDecisionVariables_numpy()
        # make clusters
        self.getClusteringResults()
        print("clustering finished!")
    


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

        # run KERNEL on device
        prg.rho_opencl( queue, (GLOBALSIZE,), (LOCALSIZE,),
                        d_rho, d_cords, d_wegiths,
                        # algorithm parameters
                        n, k, self.KERNEL_R, self.KERNEL_R_NORM, self.KERNEL_R_POWER
                        )


        prg.rhoranknh_opencl(   queue, (GLOBALSIZE,), (LOCALSIZE,),
                                d_rhorank, d_nh, d_nhd, d_cords, d_rho,
                                # algorithm paramters
                                n, k, self.MAXDISTANCE
                                )

        # copy memery from device to host
        cl.enqueue_copy(queue, self.points.rho, d_rho)
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
        rho_cuda(   d_rho, d_cords, d_wegiths,
                    # algorithm parameters
                    n, k, self.KERNEL_R, self.KERNEL_R_NORM, self.KERNEL_R_POWER,
                    grid  = (int(n/blockSize)+1,1,1),
                    block = (int(blockSize),1,1) )

        rhoranknh_cuda( d_rhorank, d_nh, d_nhd, d_cords, d_rho,
                        # algorithm paramters
                        n, k, self.MAXDISTANCE,
                        grid  = (int(n/blockSize)+1,1,1),
                        block = (int(blockSize),1,1) )

        # copy memery from device to host
        cuda.memcpy_dtoh(self.points.rho,d_rho)
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
        


    def getDecisionVariables_numpy(self):

        n, k = self.points.n, self.points.k

        # find rho 
        rho = []
        for i in range(n):
            dr = self.norm2Distance(self.points.cords, self.points.cords[i])
            local = (dr<self.KERNEL_R)
            irho = np.sum( self.points.weights[local] * np.exp( - (dr[local]/self.KERNEL_R_NORM)**self.KERNEL_R_POWER ))
            irho = np.sum( self.points.weights[local] )
            rho.append(irho)

        rho = np.array(rho)

        # find rhorank
        argsortrho = rho.argsort(kind='mergesort')[::-1]
        rhorank = np.empty(rho.size, int)
        rhorank[argsortrho] = np.arange(rho.size)

        # find NearstHiger and distance to NearestHigher
        nh,nhd = [],[]
        for i in range(n):
            irho  = rho[i]
            irank = rhorank[i]
            
            higher = rhorank<irank
            # if no points is higher
            if not (True in higher): 
                nh. append(i)
                nhd.append(self.MAXDISTANCE)
            else:
                drr  = self.norm2Distance(self.points.cords[higher], self.points.cords[i])
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

    def norm2Distance(self, p1,p2):
        return np.sqrt( np.sum((p1-p2)**2, axis=-1) )