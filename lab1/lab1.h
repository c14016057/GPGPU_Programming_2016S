#pragma once
#include <cstdint>
#include <memory>
#include <stdint.h>
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 960;
using std::unique_ptr;

struct Lab1VideoInfo {
	unsigned w, h, n_frame;
	unsigned fps_n, fps_d;
};

class Lab1VideoGenerator {
	struct Impl;
	unique_ptr<Impl> impl;
public:
	Lab1VideoGenerator();
	~Lab1VideoGenerator();
	void get_info(Lab1VideoInfo &info);
	void Generate(uint8_t *yuv);
	void YUVconvertor();
	uint8_t r[W*H], g[W*H], b[W*H];
	uint8_t y[W*H], u[W*H/4], v[W*H/4];
};
