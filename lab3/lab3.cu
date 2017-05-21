#include "lab3.h"
#include <cstdio>
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
__device__ float TARGETCOLOR(	
	const float *target,
	const int xt, const int yt, const int color, const int oldx, const int oldy,
	const int wt, const int ht
) {
	if(0 <= xt && xt < wt && 0 <= yt && yt < ht)
		return target[(wt*oldy+oldx)*3 + color] - target[(wt*yt+xt)*3 + color];
	else
		return 0;	
}



__device__  float FIX(
	const float *target,
	const int xt, const int yt, const int color, 
	const int wt, const int ht
) {
	return (TARGETCOLOR(target,xt-1, yt, color, xt, yt, wt, ht) + 
			TARGETCOLOR(target,xt, yt-1, color, xt, yt, wt, ht) + 
			TARGETCOLOR(target,xt+1, yt, color, xt, yt, wt, ht) + 
			TARGETCOLOR(target,xt, yt+1, color, xt, yt, wt, ht));
}



__device__ float findBackground(const float *background,const int color,  const int xt, const int yt, const int wb, const int hb, const int ox, const int oy) {
	int safex = xt + ox, safey = yt + oy;
	safex = safex < 0 ? 0 : (safex >= wb ? wb-1 : safex);

	safey = safey < 0 ? 0 : safey;
	safey = safey >= hb ? hb-1 : safey;

	return background[(safey * wb + safex)*3 + color];

}


__device__ float BUFFERCOLOR(	
	const float *source,
	const float *background,
	const float *mask,
	const int xt, const int yt, const int color, 
	const int wt, const int ht, const int wb, const int hb, const int ox, const int oy
) {
	if(0<=yt && yt < ht && 0 <= xt && xt < wt) {
		//INMASK
		if( mask[wt*yt+xt] > 127.0f ) {
			return source[(wt*yt+xt)*3 + color];
		//OUTMASK
		} else {
			return findBackground(background , color, xt, yt, wb, hb, ox, oy);
		}
	//OUT TARGET
	} else {
		return findBackground(background, color, xt, yt ,wb, hb, ox, oy);
	}
}


__device__ float BUFFER(
	const float *source,
	const float *background,
	const float *mask,
	const int xt, const int yt, const int color,
	const int wt, const int ht, const int wb, const int hb, const int ox, const int oy
) {
	return  BUFFERCOLOR(source, background , mask, xt-1, yt, color, wt, ht, wb, hb, ox, oy) +
			BUFFERCOLOR(source, background , mask, xt, yt-1, color, wt, ht, wb, hb, ox, oy) +
			BUFFERCOLOR(source, background , mask, xt+1, yt, color, wt, ht, wb, hb, ox, oy) +
			BUFFERCOLOR(source, background , mask, xt, yt+1, color, wt, ht, wb, hb, ox, oy); 
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
	if (0 <= yt && yt < ht && 0 <= xt && xt < wt) {
		dest[curt*3+0] = (FIX(target, xt, yt, 0, wt, ht) 
						+ BUFFER(source, background, mask, xt, yt , 0, wt, ht, wb, hb, ox, oy))/4.0f;
		dest[curt*3+1] = (FIX(target, xt, yt, 1, wt, ht) 
						+ BUFFER(source, background, mask, xt, yt , 1, wt, ht, wb, hb, ox, oy))/4.0f;
		dest[curt*3+2] = (FIX(target, xt, yt, 2, wt, ht) 
						+ BUFFER(source, background, mask, xt, yt , 2, wt, ht, wb, hb, ox, oy))/4.0f;
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
