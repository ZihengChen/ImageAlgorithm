import pyopencl as cl

def openclKernel(DeviceID=0):

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[DeviceID]
    
    context = cl.Context([device])

    #print("Device: {}".format(device.name))
    #print("Device MaxWorkGroupSize: {}".format(device.max_work_group_size))
    
    
    prg = cl.Program(context,"""

    // 1. get rho
     
    __kernel void rho_opencl(   __global float *d_rho,
                                __global float *d_Points, __global float *d_wPoints,
                                int nPoints, int kPoints, float KERNEL_R, float KERNEL_R_NORM, float KERNEL_R_POWER
                                )
    {
        int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
        int idx_point = i * kPoints;

        if( i < nPoints ) {
            
            // const int dim = kPoints;
            float pointi[2];
            // float * pointi;

            for (int k = 0; k<kPoints; k++){
                pointi[k] = d_Points[idx_point + k];
            }
    

            // loop over all points to calculate rho
            float rhoi = 0.0;
            
            for (int j=0; j<nPoints; j++){

                float dr = 0;

                for (int k = 0; k < kPoints; k++){
                    dr += pow( d_Points[ j*kPoints + k] - pointi[k] , 2) ;
                }
                dr = sqrt(dr);
                
                if (dr<KERNEL_R){
                    // float expWeight = exp(- pow(dr/KERNEL_R_NORM, KERNEL_R_POWER) );
                    rhoi += d_wPoints[j] ;

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
    
    
    
    // 2. get rhorank and nh+nhd 2in1

    __kernel void rhoranknh_opencl( __global int *d_rhorank, __global int *d_nh, __global float *d_nhd,
                                    // input parameters
                                    __global float *d_Points, __global float *d_rho,
                                    int nPoints, int kPoints, float MAXDISTANCE
                                    )
    {
        int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
        int idx_point = i * kPoints;

        if( i < nPoints ) {

            // const int dim = kPoints;
            float pointi[2];
            //float * pointi;

            for (int k = 0; k<kPoints; k++){
                pointi[k] = d_Points[idx_point + k];
            }

            float rhoi = d_rho[i];
            
            // loop over other points to calculate rhorank, nh,nhd
            int rhoranki = 0;
            int nhi      = i;
            float nhdi   = MAXDISTANCE;
            
            for (int j=0; j<nPoints; j++){
                // calculate rhorank
                if( d_rho[j]>rhoi ) rhoranki++;
                else if ( (d_rho[j]==rhoi) && (j>i)) rhoranki++;
                
           
                // find nh and nhd
                float dr = 0;
                for (int k = 0; k < kPoints; k++){
                    dr += pow( d_Points[ j*kPoints + k] - pointi[k] , 2) ;
                }
                dr = sqrt(dr);

                // if nearer AND higher rho
                
                bool isNearer = dr<nhdi ;
                bool isHigher = d_rho[j]>rhoi || (d_rho[j]==rhoi && j>i) ;
                if ( isNearer && isHigher ){ 
                    nhdi = dr;
                    nhi  = j;
                }
            }
            
            d_rhorank[i] = rhoranki;
            d_nh[i]      = nhi;
            d_nhd[i]     = nhdi;
        }
    }

    """
    ).build()
    return context,prg


