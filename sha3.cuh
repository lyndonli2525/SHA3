

#ifndef SHA3_H
#define SHA3_H

#include <stddef.h>
#include <stdint.h>
#include <cuda.h>




#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))
typedef unsigned char BYTE;


typedef struct {
    union {                                 
        uint8_t b[200];       //8 bit bytes             
        uint64_t q[25];       //64 bit words              
    } A;
    int pt; 
	int rsiz; 
	int mdlen;                    
} sha3_ctx;

typedef struct JOB {
	size_t size;
	uint8_t digest[64];
	char * data;
}JOB;

void pre_sha3();
__device__ void sha3_transform(uint64_t st[25]);


__device__ void sha3_init(sha3_ctx *ctx);    
__device__ void sha3_update(sha3_ctx *ctx, const void *data, size_t len);
__device__ void sha3_final(sha3_ctx *ctx, void *md);    

__global__ void sha3(const void *in, int n, size_t inlen, void *md);
void print_job(JOB **jobs, int num_jobs);
void runJobs(JOB** jobs, int num_inputs);
int getNumInp(const char *inpF);


#endif