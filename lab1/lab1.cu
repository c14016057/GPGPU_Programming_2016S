#include "lab1.h"
#include <stdlib.h>
#include <time.h>
#define L 1024
struct Lab1VideoGenerator::Impl {
	int t = 0;
};
int dist(int ai, int aj, int bi, int bj) {
	return (ai-bi)*(ai-bi)+(aj-bj)*(aj-bj);
//	return abs(ai-bi)+abs(aj-bj);
}
int li[L]={}, lj[L]={}; 
double nowL = 2;
uint8_t lr[L]={}, lg[L]={}, lb[L]={};
void Kmean(uint8_t *r, uint8_t *g, uint8_t *b) {
	int lsi[L] = {}, lsj[L] = {}, lcount[L] = {};
	for(int i = 0; i < H; i++)
		for(int j = 0; j < W; j++) {
			int leader = 0;
			for(int l = 1; l < nowL; l++) {
				if(dist(i,j,li[l],lj[l])<dist(i,j,li[leader],lj[leader])) leader=l;
			}
			r[i*W+j]=lr[leader];
			g[i*W+j]=lg[leader];
			b[i*W+j]=lb[leader];
			lsi[leader] += i;
			lsj[leader] += j;
			lcount[leader] ++;
		}
	for(int l = 0; l < nowL; l++) {
		li[l] = (9*li[l] + lsi[l] / lcount[l])/10;
		lj[l] = (9*lj[l] + lsj[l] / lcount[l])/10;
	}
}
Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
	srand(time(NULL));
	for (int i = 0; i < H; i++)
		for (int j = 0; j < W; j++) {
			r[i*W+j]=0;
			g[i*W+j]=0;
			b[i*W+j]=0;
		}
	for(int l = 0; l < L; l++) {
		li[l] = rand()%H;
		lj[l] = rand()%W;
		lr[l] = rand()%256;
		lg[l] = rand()%256;
		lb[l] = rand()%256;
	}
/*	for(int i = 0; i < 1000;i++) {
		int idi = rand()%(H/4);
		int idj = rand()%(W/4);
		for(int rr = 0; rr < 4; rr++) {
			for(int dd = 0; dd < 4; dd++) {
				r[(4*idi+dd)*W+idj*4+rr]=0;
				g[(4*idi+dd)*W+idj*4+rr]=255;
				b[(4*idi+dd)*W+idj*4+rr]=0;
			}
		}
	}*/	
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	Kmean(r, g, b);
	YUVconvertor();
	if(nowL < L) nowL +=0.25;
	cudaMemcpy(yuv, y, W*H, cudaMemcpyHostToDevice);
	cudaMemcpy(yuv+W*H, u, W*H/4,cudaMemcpyHostToDevice);
	cudaMemcpy(yuv+W*H*5/4, v, W*H/4,cudaMemcpyHostToDevice);
}
void Lab1VideoGenerator::YUVconvertor() {
	for(int i = 0; i < H; i++)
		for(int j = 0; j < W; j++)
			y[i*W+j] = (uint8_t)(0.299*r[i*W+j] + 0.587*g[i*W+j] + 0.114*b[i*W+j]);
/*	for(int i = 0; i < H; i++)
		for(int j = 0; j < W; j += 4) {
			double r4 = (r[i*W+j]+r[i*W+j+1]+r[i*W+j+2]+r[i*W+j+3])/4;
			double g4 = (g[i*W+j]+g[i*W+j+1]+g[i*W+j+2]+g[i*W+j+3])/4;
			double b4 = (b[i*W+j]+b[i*W+j+1]+b[i*W+j+2]+b[i*W+j+3])/4;
			u[i*W/4+j/4] = (uint8_t)(-0.169*r4 - 0.331*g4 + 0.5*b4 +128);
			v[i*W/4+j/4] = (uint8_t)(0.5*r4 - 0.419*g4 - 0.081*b4 +128);}*/
	for(int i = 0; i < H; i +=2)
		for(int j = 0; j < W; j += 2) {
			double r4 = (r[i*W+j]+r[i*W+j+1]+r[(i+1)*W+j]+r[(i+1)*W+j+1])/4;
			double g4 = (g[i*W+j]+g[i*W+j+1]+g[(i+1)*W+j]+g[(i+1)*W+j+1])/4;
			double b4 = (b[i*W+j]+b[i*W+j+1]+b[(i+1)*W+j]+b[(i+1)*W+j+1])/4;
			u[(i/2)*W/2+j/2] = (uint8_t)(-0.169*r4 - 0.331*g4 + 0.5*b4 +128);
			v[(i/2)*W/2+j/2] = (uint8_t)(0.5*r4 - 0.419*g4 - 0.081*b4 +128);}
}
