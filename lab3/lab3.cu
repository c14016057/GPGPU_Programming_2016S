#include "lab3.h"
#include <cstdio>
#define INMASK(xt, yt) ( (!(0 <=yt && yt < ht && 0 <= xt && xt < wt)) ? false : (mask[wt*(yt)+xt] > 127.0f) )
#define TARGETCOLOR(xt, yt, color, oldx, oldy) ( (0<=(xt) && (xt) < wt && 0 <= (yt) && (yt) < ht) ? target[(wt*oldy+oldx)*3 + color]-target[(wt*(yt)+xt)*3+color] : \
																									0/*target[(wt*oldy+oldx)*3 + color]-255*/)
#define BUFFERCOLOR(xt, yt, color, oldx, oldy) ( (!(0 <=(yt) && (yt) < ht && 0 <= (xt) && (xt) < wt)) ? ( 0<=(yt)+oy && (yt)+oy < hb && 0<=xt+ox && xt+ox < wb ? \
									background[(((yt)+oy)*wb+(xt)+ox)*3+color] : background[(((oldy)+oy)*wb+(oldx)+ox)*3+color])\
									 : ( mask[wt*(yt)+(xt)] > 127.0f ? ( 0<=(yt)+oy && (yt)+oy < hb && 0<=xt+ox && xt+ox < wb ? \
									 source[(wt*(yt)+(xt))*3+color] : background[(wb*(oldy+oy)+(oldx)+ox)*3+color]) : ( 0<=(yt)+oy && (yt)+oy < hb && 0<=xt+ox && xt+ox < wb ? background									[(((yt)+oy)*wb+(xt)+ox)*3+color]: background[(((oldy)+oy)*wb+(oldx)+ox)*3+color])))
#define FIX(xt, yt, color) (TARGETCOLOR(xt-1, yt, color, xt, yt) + \
							TARGETCOLOR(xt, yt-1, color, xt, yt) + \
							TARGETCOLOR(xt+1, yt, color, xt, yt) + \
							TARGETCOLOR(xt, yt+1, color, xt, yt))
#define BUFFER(xt, yt, color) ( BUFFERCOLOR(xt-1, yt, color, xt, yt) + \
								BUFFERCOLOR(xt, yt-1, color, xt, yt) + \
								BUFFERCOLOR(xt+1, yt, color, xt, yt) + \
								BUFFERCOLOR(xt, yt+1, color, xt, yt))
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}
__global__ void PoissonImageCloningIteration(
	const float *background,
	const float *target,
	const float *mask,
	float *source,
	float *dest,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if ( true/*INMASK(xt, yt)  yt < ht and xt < wt and mask[curt] > 127.0f*/ ) { 	
		const int yb = oy+yt, xb = ox+xt;
		//const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			dest[curt*3+0] = (FIX(xt, yt, 0) + BUFFER(xt, yt , 0))/4.0f;
			dest[curt*3+1] = (FIX(xt, yt, 1) + BUFFER(xt, yt , 1))/4.0f;
			dest[curt*3+2] = (FIX(xt, yt, 2) + BUFFER(xt, yt , 2))/4.0f;
		}
	} 
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
) {
	// set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	// initialize the iteration
	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	/*CalculateFixed<<<gdim, bdim>>>(
	background, target, mask, fixed,
	wb, hb, wt, ht, oy, ox
	);*/
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	// iterate
	for (int i = 0; i < 10000; ++i) {
	
	PoissonImageCloningIteration<<<gdim, bdim>>>(
	background, target, mask, buf1, buf2, wb, hb, wt, ht, oy, ox
	);

	PoissonImageCloningIteration<<<gdim, bdim>>>(
	background, target, mask, buf2, buf1, wb, hb, wt, ht, oy, ox
	);
	}

//	PoissonImageCloningIteration<<<gdim, bdim>>>(
//	background, target, mask, buf1, buf2, wb, hb, wt, ht, oy, ox
//	);
	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<gdim, bdim>>>(
	background, buf1, mask, output,
	wb, hb, wt, ht, oy, ox
	);

	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
