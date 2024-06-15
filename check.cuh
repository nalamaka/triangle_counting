#ifndef __CUDA_CHECK__
#define __CUDA_CHECK__
#include<stdio.h>
#include<cuda_runtime.h>
#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if(error != cudaSuccess)                                                    \
    {                                                                           \
        printf("Error: %s:%d, ", __FILE__,__LINE__);                            \
        printf("code:%d,reason:%d\n",error,cudaGetErrorString(error));          \
        exit(1);                                                                \
    }                                                                           \
}
#endif