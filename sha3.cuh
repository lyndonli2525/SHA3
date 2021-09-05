

#ifndef SHA3_H
#define SHA3_H
#include <stddef.h>
#include <stdint.h>
#include <cuda.h>

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

//SHA3 context 
typedef struct {
	union {
	  uint8_t b[200];		//8 bit bytes
	  uint64_t q[25];		//64 bit words
	}A;
	int pt;
	int rsiz;
	int mdlen;
} sha3_ctx;

//Structure to hold messages
typedef struct JOB {
	size_t size;
	uint8_t digest[64];
	char * data;
}JOB;

//copies sha3 constants to host
void pre_sha3();

//Updates SHA3 context's state
__device__ void sha3_transform(uint64_t st[25]);

//Initiates SHA3 context
__device__ void sha3_init(sha3_ctx *ctx);

//Absorbs input blocks for SHA3
__device__ void sha3_update(sha3_ctx *ctx, const void *data, size_t len);

//Finalizes SHA3 hash output
__device__ void sha3_final(sha3_ctx *ctx, void *md);

//Runs SHA3 function across a single line
__global__ void sha3(const void *in, int n, size_t inlen, void *md);

//Prints out SHA3 outputs
void print_job(JOB **jobs, int num_jobs);

//Runs SHA3 using GPU
void runJobs(JOB** jobs, int num_inputs);

//Gets number of inputs in file
int getNumInp(const char *inpF);

#endif