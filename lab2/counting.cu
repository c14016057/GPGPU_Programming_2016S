#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void yuan(const char *text, int *pos, int text_size) {
	int textP = blockIdx.x * blockDim.x + threadIdx.x;
	if (textP >= text_size) return;
	const char *start = text + textP;
	while (start >= text && *start > ' ') {
		start--;
	}
	pos[textP] = text + textP - start;

}
struct to_key 
{
	__host__ __device__ int operator()(char c) 
	{
		return c <= ' '? 0 : 1;
	}
};	

void CountPosition1(const char *text, int *pos, int text_size)
{
	thrust :: transform (thrust :: device, text, text + text_size, pos, to_key());
	thrust :: inclusive_scan_by_key (thrust :: device, pos, pos + text_size, pos, pos);
}

void CountPosition2(const char *text, int *pos, int text_size)
{

	yuan<<< (text_size/1024 + 1) , (1<<10) >>>(text, pos, text_size);
}
