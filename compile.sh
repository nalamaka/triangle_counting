nvcc -gencode arch=compute_70,code=sm_70 -Xptxas -v -DUSE_GPU -O3 -w -DEDGE_PAR  test_tc.cu -o test_tc
