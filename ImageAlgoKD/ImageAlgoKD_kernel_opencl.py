import pyopencl as cl

def openclKernel(DeviceID=0):

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[DeviceID]
    
    context = cl.Context([device])

    #print("Device: {}".format(device.name))
    #print("Device MaxWorkGroupSize: {}".format(device.max_work_group_size))
    
    
    prg = cl.Program(context,"""

    // 1.1 get rho
     
    __kernel void rho_opencl(   __global float *d_rho,
                                __global float *d_Points, __global float *d_wPoints,
                                const int nPoints, const int kPoints, float KERNEL_R, float KERNEL_R_NORM, float KERNEL_R_POWER
                                )
    {
        int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
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
    __kernel void rho_openclbin(__global float *d_rho,
                                __global float *d_Points, __global float *d_wPoints,
                                __global int *d_nnbinHead,
                                __global int *d_nnbinSize,
                                __global int *d_nnbinList,
                                __global int *d_idxPointsHead, 
                                __global int *d_idxPointsSize,
                                __global int *d_idxPonitsList,
                                const int nPoints, const int kPoints, float KERNEL_R, float KERNEL_R_NORM, float KERNEL_R_POWER
                                )
    {
        int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
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

    __kernel void rhoranknh_opencl( __global int *d_rhorank, __global int *d_nh, __global float *d_nhd,
                                    // input parameters
                                    __global float *d_Points, __global float *d_rho,
                                    const int nPoints, int kPoints, float MAXDISTANCE
                                    )
    {
        int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
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
    
    ).build()
    return context,prg


