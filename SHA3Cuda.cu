#include "sha3.cuh"
#include <string.h>
#include <stdio.h>
//references
//https://github.com/XKCP/XKCP/blob/master/Standalone/CompactFIPS202/C/Keccak-readable-and-compact.c
//https://keccak.team/keccak_specs_summary.html
__constant__ uint64_t dev_keccakf_rndc[24];
__constant__ int dev_keccakf_rotc[24];
__constant__ int dev_keccakf_piln[24];

//keccak round constants
static uint64_t host_keccakf_rndc[24] = {
  0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
  0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
  0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
  0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
  0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
  0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
  0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
  0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

//keccak rotational constants
static int host_keccakf_rotc[24] = {
  1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
  27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

//keccak constants for pi step   
static int host_keccakf_piln[24] = {
  10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
  15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

//update state
__device__ void sha3_transform(uint64_t A[25])
{
  int i, j, r;
  uint64_t D, C[5];
  for (r = 0; r < 24; r++) {
  // Theta
    for (i = 0; i < 5; i++)
      C[i] = A[i] ^ A[i + 5] ^ A[i + 10] ^ A[i + 15] ^ A[i + 20];
      for (i = 0; i < 5; i++) {
        D = C[(i + 4) % 5] ^ ROTL64(C[(i + 1) % 5], 1);
        for (j = 0; j < 25; j += 5)
          A[j + i] ^= D;
  }

  // Rho Pi
  D = A[1];
  for (i = 0; i < 24; i++) {
    j = dev_keccakf_piln[i];
    C[0] = A[j];
    A[j] = ROTL64(D, dev_keccakf_rotc[i]);
    D = C[0];
  }

  //  Chi
  for (j = 0; j < 25; j += 5) {
    for (i = 0; i < 5; i++)
      C[i] = A[j + i];
      for (i = 0; i < 5; i++)
        A[j + i] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
  }
  // Iota
  A[0] ^= dev_keccakf_rndc[r];
  }
}

//initialize sha3 context
__device__ void sha3_init(sha3_ctx *ctx)
{
  for (int i = 0; i < 25; i++) {
    ctx->A.q[i] = 0;
  }
  ctx->mdlen = 32;
  ctx->rsiz = (200 - (2 * 32));
  ctx->pt = 0;
}

//absorb input blocks
__device__ void sha3_update(sha3_ctx *ctx, const void *data, size_t len)
{
  size_t i;
  int j;
  j = ctx->pt;
  for (i = 0; i < len; i++) {
    ctx->A.b[j++] = ((const uint8_t *) data)[i];
    if (j >= ctx->rsiz) {
      sha3_transform(ctx->A.q);
      j = 0;
    }
  }
  ctx->pt = j;
}

//add delimiter suffix to end of hash
__device__ void sha3_final(sha3_ctx *ctx, void *md)
{
  ctx->A.b[ctx->pt] = 0x06;
  ctx->A.b[ctx->rsiz - 1] = 0x80;
  sha3_transform(ctx->A.q);
  for (int i = 0; i < ctx->mdlen; i++) {
    ((uint8_t *) md)[i] = ctx->A.b[i];
  }
}

//parallel computing of multiple inputs from text file
__global__ void sha3(JOB **jobs, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    sha3_ctx ctx;
    sha3_init(&ctx);
    sha3_update(&ctx, jobs[idx]->data, jobs[idx]->size);
    sha3_final(&ctx, jobs[idx]->digest);
  }
}

//Run SHA3 function for GPU
void runJobs(JOB **jobs, int num_jobs)
{
  int blockSize = 4;
  int numBlocks = (num_jobs + blockSize - 1) / blockSize;
  sha3 <<< numBlocks, blockSize >>> (jobs, num_jobs);
}

//Copies constants to host device
void pre_sha3() {
  cudaMemcpyToSymbol(dev_keccakf_rndc, host_keccakf_rndc, sizeof(host_keccakf_rndc), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_keccakf_rotc, host_keccakf_rotc, sizeof(host_keccakf_rotc), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dev_keccakf_piln, host_keccakf_piln, sizeof(host_keccakf_piln), 0, cudaMemcpyHostToDevice);
}

//print out hash values
void print_job(JOB **jobs, int num_jobs) {
  for (int i=0; i<num_jobs; i++) {
    for(int j=0; j<jobs[i]->size; j++) {
      printf("%c", jobs[i]->data[j]);
    }
    printf("\n");
    for(int j=0; j<32; j++) {
      printf("%02x", jobs[i]->digest[j]);
    }
    printf("\n");
  }
}

//get number of inputs in text file
int getNumInp(const char *inpF){
  FILE *inp;
  inp = fopen(inpF, "r"); // filename of your data file
  size_t num_inputs = 0;
  while (1) {
    char r = (char)fgetc(inp);
    while (((r != ' ') && (r!= '\n')) && !feof(inp)) { // read till , or EOF
      r = (char)fgetc(inp);
    }
    if (feof(inp)) { 
      break;
    }
    num_inputs++;
  }
  fclose(inp);
  return num_inputs;
}

