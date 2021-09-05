#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "sha3.cuh"

//Funtion to run SHA3 function across input file
void gpu_Run(const char *inpF){
  JOB **jobs = NULL;
  FILE *inp;
  if(inp = fopen(inpF, "r")) { // filename of your data file
    size_t num_inputs = getNumInp(inpF);
    cudaMallocManaged(&jobs, num_inputs * sizeof(JOB*));
    int i = 0;
    while (1) {
      int size = 0;
      char r = (char)fgetc(inp);
      while (r!= '\n' && r != ' ' && !feof(inp)) { // read till , or EOF
        size++;
        r = (char)fgetc(inp);
      }
      if (size!=0) {//create jobs with size of inputs
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
        buf[count] = r;// store characters in array
        r = (char)fgetc(inp);
        count++;
      }
        buf[count] = '\0';
      if (feof(inp)) { // check again for EOF
        break;
      }
      jobs[i]->data = buf;
    }

    //Prepare the SHA3 constants
    pre_sha3();

    //Generate hash outputs
    runJobs(jobs, num_inputs);

    //Synchronize the threads
    cudaDeviceSynchronize();

    //Print out the hash outputs
    print_job(jobs, num_inputs);

    //Remove device allocations
    cudaDeviceReset();
  }
  else {
    printf("File doesn't exist.");
  }
}
int main(int argc, char **argv)
{
  //ask for filename
  char str[20];
  printf("Please enter the filename:\n");
  scanf("%s", str);

  //run sha3 in cuda
  gpu_Run(str);
  return 0;
}