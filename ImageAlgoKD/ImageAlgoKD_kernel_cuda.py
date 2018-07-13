## cuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""

    // 1. get rho

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
rhoranknh_cuda = mod.get_function("rhoranknh_cuda")
