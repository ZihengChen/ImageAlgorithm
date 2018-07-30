## cuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""

    // 1.1 get rho

    __global__ void rho_cuda( float *d_rho, 
                              float *d_Points, float *d_wPoints,
                              int nPoints, int kPoints, float KERNEL_R, float KERNEL_R_NORM, float KERNEL_R_POWER){

        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int idx_point = i * kPoints;

        if( i < nPoints ) {
            
            float rhoi = 0.0;

            // loop over all points to calculate rho
            for (int j=0; j<nPoints; j++){

                float dr = 0;

                for (int k = 0; k < kPoints; k++){
                    dr += pow( d_Points[ j*kPoints + k] - d_Points[idx_point + k] , 2) ;
                }
                dr = sqrt(dr);
                
                if (dr<KERNEL_R){
                    float expWeight = 1.0;
                    if (KERNEL_R_POWER != 0)
                        expWeight = exp(- pow(dr/KERNEL_R_NORM, KERNEL_R_POWER) );
                    rhoi += d_wPoints[j] * expWeight;

                    ///////////////////////////////////////////////////
                    // some device does not support exp() function   //
                    // have to use Tylor expansion for exp() instead //
                    ///////////////////////////////////////////////////

                    // float d = pow(dr/KERNEL_R_NORM, KERNEL_R_POWER);
                    // float expWeight = 1 / (1 + d + d*d/2 + d*d*d/6 + d*d*d*d/24);
                    // rhoi += d_wPoints[j] * expWeight
                }
            }

            d_rho[i] = rhoi;
        }
    }


    // 1.2 get rho with NNBin
    __global__ void rho_cudabin(float *d_rho, 
                                float *d_Points, float *d_wPoints,
                                int *d_nnbinHead,
                                int *d_nnbinSize,
                                int *d_nnbinList,
                                int *d_idxPointsHead, 
                                int *d_idxPointsSize,
                                int *d_idxPonitsList,
                                int nPoints, int kPoints, float KERNEL_R, float KERNEL_R_NORM, float KERNEL_R_POWER){

        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int idx_point = i * kPoints;

        if( i < nPoints ) {
            
            float rhoi = 0.0;

            int a1 = d_nnbinHead[i];
            int b1 = a1 + d_nnbinSize[i];

            // loop over nn bins
            for (int l1=a1; l1<b1; l1++){
                int idxNNBin = d_nnbinList[l1];
                
                int a2 = d_idxPointsHead[idxNNBin];
                int b2 = a2 + d_idxPointsSize[idxNNBin];
                // loop over points in the nn bin
                for (int l2 = a2; l2<b2; l2++){
                    // get j
                    int j = d_idxPonitsList[l2];

                    float dr = 0;
                    // get dij
                    for (int k = 0; k < kPoints; k++){
                        dr += pow( d_Points[ j*kPoints + k] - d_Points[idx_point + k] , 2) ;
                    }
                    dr = sqrt(dr);
                    // get density
                    if (dr<KERNEL_R){
                        float expWeight = 1.0;
                        if (KERNEL_R_POWER != 0)
                            expWeight = exp(- pow(dr/KERNEL_R_NORM, KERNEL_R_POWER) );
                        rhoi += d_wPoints[j] * expWeight;
                        
                        ///////////////////////////////////////////////////
                        // some device does not support exp() function   //
                        // have to use Tylor expansion for exp() instead //
                        ///////////////////////////////////////////////////

                        // float d = pow(dr/KERNEL_R_NORM, KERNEL_R_POWER);
                        // float expWeight = 1 / (1 + d + d*d/2 + d*d*d/6 + d*d*d*d/24);
                        // rhoi += d_wPoints[j] * expWeight
                    }
                }
            }
            
            d_rho[i] = rhoi;
        }
    }



    // 2. get rhorank and nh+nhd 2in1

    __global__ void rhoranknh_cuda( int *d_rhorank, int *d_nh, float *d_nhd,
                                    float *d_Points, float *d_rho,
                                    int nPoints, int kPoints, float MAXDISTANCE){

        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int idx_point = i * kPoints;

        if( i < nPoints ) {

            float rhoi   = d_rho[i];
            int rhoranki = 0;
            int nhi      = i;
            float nhdi   = MAXDISTANCE;
            
            // loop over other points to calculate rhorank, nh,nhd
            for (int j=0; j<nPoints; j++){
                // calculate rhorank
                if( d_rho[j]>rhoi )
                    rhoranki++;

                else if ( (d_rho[j]==rhoi) && (j>i)) 
                    // if same rho, by definition, larger index has higher rho
                    rhoranki++;
                
                
                // find nh and nhd
                
                // if higher, larger index has higher rho
                bool isHigher = d_rho[j]>rhoi || (d_rho[j]==rhoi && j>i) ; 
                if (isHigher){
                    float dr = 0;
                    for (int k = 0; k < kPoints; k++){
                        dr += pow( d_Points[ j*kPoints + k] - d_Points[ idx_point + k] , 2 ) ;
                    }
                    dr = sqrt(dr);

                    // if nearer                
                    if ( dr<nhdi ){ 
                        nhdi = dr;
                        nhi  = j;
                    }
                }
            
            }

            d_rhorank[i] = rhoranki;
            d_nh[i]      = nhi;
            d_nhd[i]     = nhdi;        
        }
    }

"""
)

rho_cuda       = mod.get_function("rho_cuda")
rho_cudabin    = mod.get_function("rho_cudabin")
rhoranknh_cuda = mod.get_function("rhoranknh_cuda")

