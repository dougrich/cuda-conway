#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <random>

#include "../gif-h/gif.h"

#define PIXEL_DEPTH 4
#define BLOCK_SIZE 256
#define DELAY 10
#define SIMULATION_STEPS 100
#define SIMULATION_SEED 0xFFAABBEE
#define SIMULATION_SIZE 256

__device__
unsigned idx(int u, int v, unsigned size) {
  u = (u + size) % size;
  v = (v + size) % size;
  return v * size + u;
}

__device__
void set(uint8_t *w, int i, bool alive) {
  uint8_t color = alive ? 255 : 0;
  w[i * PIXEL_DEPTH + 0] = color;
  w[i * PIXEL_DEPTH + 1] = color;
  w[i * PIXEL_DEPTH + 2] = color;
  w[i * PIXEL_DEPTH + 3] = color;
}

__global__
void gameOfLifeStep(unsigned n, uint8_t *r, uint8_t *w, unsigned size) {
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    int v = i / size;
    int u = i - v * size;

    unsigned neighbors = 0;
    for (int dv = -1; dv <= 1; dv++) {
      for (int du = -1; du <= 1; du++) {
        if (r[idx(u + du, v + dv, size) * PIXEL_DEPTH] == UINT8_MAX) {
          neighbors++;
        }
      }
    }

    bool alive = r[i * PIXEL_DEPTH] == UINT8_MAX;

    set(w, i, alive ? neighbors == 2 || neighbors == 3 : neighbors == 3);
  }
}

int main(int argc, char *argv[]) {

  std::mt19937 generator (SIMULATION_SEED);

  unsigned pixCount = SIMULATION_SIZE * SIMULATION_SIZE;

  // use two buffers to avoid contention
  uint8_t *buffer[2];

  // allocate memory
  cudaMallocManaged(&buffer[0], PIXEL_DEPTH * pixCount * sizeof(uint8_t));
  cudaMallocManaged(&buffer[1], PIXEL_DEPTH * pixCount * sizeof(uint8_t));

  // init the buffers
  for (int j = 0; j < pixCount; j++) {
    unsigned color = generator() % 2 == 1
      ? UINT8_MAX
      : 0;
    for (int k = 0; k < PIXEL_DEPTH; k++) {
      buffer[0][j * PIXEL_DEPTH + k] = color;
    }
  }

  GifWriter gif;

  GifBegin(&gif, "out.gif", SIMULATION_SIZE, SIMULATION_SIZE, DELAY);

  unsigned current = 0;
  for (unsigned i = 0; i < SIMULATION_STEPS; i++) {
    // current and next are the two buffers; we always write to the other buffer to avoid read-write contention

    unsigned next = (current + 1) % 2;

    // write out the frame
    GifWriteFrame(&gif, buffer[current], SIMULATION_SIZE, SIMULATION_SIZE, DELAY);

    // simulate
    gameOfLifeStep<<<(pixCount + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(pixCount, buffer[current], buffer[next], SIMULATION_SIZE);

    // synchronize
    cudaDeviceSynchronize();

    // swap buffers
    current = next;
  }

  GifWriteFrame(&gif, buffer[current], SIMULATION_SIZE, SIMULATION_SIZE, DELAY);

  GifEnd(&gif);

  return 0;
}