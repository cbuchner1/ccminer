# possible additional CUDA_CFLAGS
#-gencode=arch=compute_50,code=\"sm_50,compute_50\"
#-gencode=arch=compute_35,code=\"sm_35,compute_35\"
#-gencode=arch=compute_30,code=\"sm_30,compute_30\"

#--ptxas-options=\"-v -dlcm=cg\""

CUDA_CFLAGS="-O3" ./configure "CFLAGS=-O3" "CXXFLAGS=-O3" --with-cuda=/usr/local/cuda

