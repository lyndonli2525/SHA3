

#include <stdio.h>
#include <string.h>
#include "sha3.cuh"

JOB ** gpu_Run(const char *inpF){
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
        rewind(inp);
        int *size_arr = (int*) malloc(num_inputs * sizeof(int));
        int i = 0;
        while (1) {
          int size = 0;
          char r = (char)fgetc(inp);
          while (r!= '\n' && r != ' ' && !feof(inp)) { // read till , or EOF
            size++;               // store in array
            r = (char)fgetc(inp);
          }
          if (size!=0) {
           size_arr[i] = size;
           size = 0;
          }
          if (feof(inp)) { // check again for EOF
            break;
          }
          i++;
        }
        
        
        rewind(inp);
        JOB **jobs = NULL;
        printf("Number of inputs: %zd\n", num_inputs);
        cudaMallocManaged(&jobs, num_inputs * sizeof(JOB*));
        for(int i=0; i<num_inputs; i++) {
          char *buf = NULL;
          JOB *job = NULL;    
          cudaMallocManaged(&buf,  size_arr[i] * sizeof(char));
          cudaMallocManaged(&job, sizeof(JOB));
          int count = 0;
          while (count < size_arr[i]) {
            char r = (char)fgetc(inp);
            while (r!= ' ' && r != '\n' && !feof(inp)) {
              buf[count] = r;               // store in array
              r = (char)fgetc(inp);
              count++;
            }
            buf[count] = '\0';
            if (feof(inp)) { // check again for EOF
              break;
            }
        }
          job->data = buf;
          job->size = size_arr[i];
          
          for(int j=0; j<64; j++)
            job->digest[j] = 0x00;
          jobs[i] = job;
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
        //char str[] = "lyndon";
        //int msg_len = strlen(str); 
        //uint8_t *buf = NULL;
//
        //cudaMallocManaged(&buf, 64);
//
        //sha3<<<1,1>>>(str,1, msg_len, buf);
        //for(int i = 0; i < 32; i++) {
        //        printf("%02x", buf[i]);
        //}
        return 0;
}