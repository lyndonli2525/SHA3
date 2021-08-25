#include <stdio.h>
#include <string.h>
#include "sha3.cuh"




JOB ** gpu_Run(const char *inpF){
  JOB **jobs = NULL;
  FILE *inp;
  inp = fopen(inpF, "r"); // filename of your data file
  size_t num_inputs = getNumInp(inpF);
  cudaMallocManaged(&jobs, num_inputs * sizeof(JOB*));
  int i = 0;
  while (1) {
    int size = 0;
    char r = (char)fgetc(inp);
    while (r!= '\n' && r != ' ' && !feof(inp)) { // read till , or EOF
      size++;               // store in array
      r = (char)fgetc(inp);
    }
    if (size!=0) {
      JOB *job = NULL;    
      cudaMallocManaged(&job, sizeof(JOB));
      job->size = size;
      jobs[i] = job;
      size = 0;
    }
    if (feof(inp)) { // check again for EOF
      break;
    }
    i++;
  } 
  rewind(inp);
  printf("Number of inputs: %zd\n", num_inputs);
  for(int i=0; i<num_inputs; i++) {
    char *buf = NULL;   
    cudaMallocManaged(&buf,  jobs[i]->size * sizeof(char));
    int count = 0;
    char r = (char)fgetc(inp);
    while (count < jobs[i]->size) {
      buf[count] = r;               // store in array
      r = (char)fgetc(inp);
      count++;
    }
      buf[count] = '\0';
    if (feof(inp)) { // check again for EOF
      break;
    }
    jobs[i]->data = buf;
  }
  pre_sha3();
  runJobs(jobs, num_inputs);
  cudaDeviceSynchronize();
  print_job(jobs, num_inputs);
  cudaDeviceReset();
  return jobs;
}
int main(int argc, char **argv)
{
        JOB** jobs = gpu_Run("test2.txt");
        return 0;
}