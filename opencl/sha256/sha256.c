/*
 * sha256.cu Implementation of SHA256 Hashing    
 *
 * Date: 12 June 2019
 * Revision: 1
 * *
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */


//This algorithm produces 256 bit hash and uses a block size of 512 bits (64 bytes)


#include<stdint.h>
#include<stdio.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#define BILLION 1000000000L
/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>

/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

/**************************** DATA TYPES ****************************/



#define MAX_SOURCE_SIZE (0x100000)
#define THREADS_PER_GROUP 256

cl_assert(cl_int  const code , char const * const file, int const line) {

	if(code!=CL_SUCCESS) 
	{
		fprintf(stderr,"\"%s\", line %d, cl_assert (%d) \n",file,line,code);
	}

}


//Difference between end times to two different 
double diff_events(cl_event event_first, cl_event event_later) {
	cl_ulong start_time, end_time;
	clGetEventProfilingInfo(event_first, CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&start_time,NULL);
	clGetEventProfilingInfo(event_later, CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end_time,NULL);
	double total_time = (end_time - start_time)* (1e-6);  //Time converted from nanoseconds to milliseconds;
	return total_time;



}

double diff_events_start_end(cl_event event_first, cl_event event_later) {
	cl_ulong start_time, end_time;
	clGetEventProfilingInfo(event_first, CL_PROFILING_COMMAND_SUBMIT,sizeof(cl_ulong),&start_time,NULL);
	clGetEventProfilingInfo(event_later, CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end_time,NULL);
	double total_time = (end_time - start_time)* (1e-6);  //Time converted from nanoseconds to milliseconds;
	return total_time;



}

double get_event_time(cl_event event) {
	cl_ulong start_time, end_time;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,sizeof(cl_ulong),&start_time,NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end_time,NULL);
	double total_time = (end_time - start_time)* (1e-6);  //Time converted from nanoseconds to milliseconds;
	return total_time;



}

#define cl(...) cl_assert((cl##__VA_ARGS__),__FILE__,__LINE__);
#define cl_ok(err) cl_assert(err,__FILE__,__LINE__);

cl_event dtrans_to_gpu;
cl_event dtrans_to_cpu;
cl_event kernel_launch;
  	


//cudaEvent_t start,stop;
//cudaEvent_t stop_dtrans_gpu;  //To track CPU to GPU data transfer
//cudaEvent_t start_dtrans_cpu; //To track GPU to CPU data transfer

double total_time;	
double total_dtrans_gpu_time;
double total_dtrans_cpu_time;
double kernel_execution_time;

struct timespec start_gpu, end_gpu;
uint64_t  total_gpu;



typedef unsigned char BYTE;
typedef unsigned int  WORD;
typedef unsigned long long LONG;


typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} CUDA_SHA256_CTX;

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
/*
__constant__ WORD k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
*/

//Start modifying from this function!!!
void mcm_cuda_sha256_hash_batch(BYTE* in, WORD inlen, BYTE* out, WORD n_batch)
{
// Load the kernel source code into the array source_str
        FILE *kernel_fp;
    	char *kernel_str;
	size_t kernel_str_size;
  	
   	kernel_fp = fopen("sha256.cl", "r");
  	if (!kernel_fp) {
        	fprintf(stderr, "Failed to load kernel.\n");
        	exit(1);
    	}
    	kernel_str = (char*)malloc(MAX_SOURCE_SIZE);
	kernel_str_size = fread( kernel_str, 1, MAX_SOURCE_SIZE, kernel_fp);
    	fclose( kernel_fp );
 
    // Get platform and device information
    	cl_platform_id platform_id = NULL;
    	cl_device_id device_id = NULL;   
    	cl_uint ret_num_devices;
    	cl_uint ret_num_platforms;
    	cl_int ret;
	cl(GetPlatformIDs(1, &platform_id, &ret_num_platforms));
    	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
            &device_id, &ret_num_devices);
	cl_ok(ret);
    		
    // Create an OpenCL context
    	cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
	cl_ok(ret);
    	
    // Create a command queue
        cl_queue_properties props[]= {CL_QUEUE_PROPERTIES,CL_QUEUE_PROFILING_ENABLE,0};
    	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &ret);
    	//cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    	cl_ok(ret);	


	clock_gettime(CLOCK_MONOTONIC,&start_gpu);

	cl_mem cuda_indata = clCreateBuffer(context, CL_MEM_READ_ONLY, inlen*n_batch*sizeof(BYTE),NULL, &ret);
cl_ok(ret);
	cl_mem cuda_outdata = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SHA256_BLOCK_SIZE*n_batch*sizeof(BYTE),NULL,&ret);
cl_ok(ret);
	cl(EnqueueWriteBuffer(command_queue,cuda_indata,CL_TRUE,0,inlen*n_batch*sizeof(BYTE),in,0,NULL,&dtrans_to_gpu));

	cl_program program = clCreateProgramWithSource(context,1,(const char **)&kernel_str, (const size_t *)&kernel_str_size, &ret);
	cl_ok(ret);
	ret= clBuildProgram(program, 1, &device_id,NULL,NULL,NULL);
cl_ok(ret);
/*

	if (ret == CL_BUILD_PROGRAM_FAILURE) {

    		// Determine the size of the log
        	size_t log_size;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	       	// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		 printf("%s\n", log);
	
	} 
*/
	cl_kernel kernel = clCreateKernel(program,"kernel_sha256_hash",&ret);
cl_ok(ret);
	cl(SetKernelArg(kernel,0,sizeof(cl_mem),(void*)&cuda_indata));
	cl(SetKernelArg(kernel,1,sizeof(WORD),(void*)&inlen));
	cl(SetKernelArg(kernel,2,sizeof(cl_mem),(void*)&cuda_outdata));
	cl(SetKernelArg(kernel,3,sizeof(WORD),(void*)&n_batch));
size_t local_item_size = THREADS_PER_GROUP;
size_t global_item_size = ((size_t)ceil((double)(n_batch)/(double)(THREADS_PER_GROUP)))*THREADS_PER_GROUP;
	cl(EnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0,NULL,&kernel_launch));
	cl(EnqueueReadBuffer(command_queue,cuda_outdata, CL_TRUE,0, SHA256_BLOCK_SIZE*n_batch*sizeof(BYTE),out,0,NULL,&dtrans_to_cpu));	


	cl(Flush(command_queue));
	cl(Finish(command_queue));

	clock_gettime(CLOCK_MONOTONIC,&end_gpu);
	
	cl(ReleaseKernel(kernel));
	cl(ReleaseProgram(program));
	cl(ReleaseMemObject(cuda_indata));
	cl(ReleaseMemObject(cuda_outdata));
	cl(ReleaseCommandQueue(command_queue));
	cl(ReleaseContext(context));

}

int main(int argc, char *argv[]) {



	printf("Usage:  ./sha256 <Filename> <total_blocks> <block_size> \n");

	unsigned int block_size, total_blocks;
	unsigned char *cpu_data;

	unsigned char *hash_cpu;
	struct timespec start_read, end_read;
	uint64_t  read_time;

	unsigned size;


	total_blocks=atoi(argv[2]);
	block_size = atoi(argv[3]); 
        size= total_blocks*block_size;
	printf("Total blocks =%u, Block size = %u , Total data =%u\n",total_blocks, block_size, size);
	cpu_data = (unsigned char*)malloc(size*sizeof(unsigned char));
	hash_cpu = (unsigned char *)malloc(total_blocks*SHA256_BLOCK_SIZE*sizeof(unsigned char));


	clock_gettime(CLOCK_MONOTONIC,&start_read);
	FILE *fptr = fopen(argv[1],"rb");
	fread(cpu_data,size,1,fptr);
	clock_gettime(CLOCK_MONOTONIC,&end_read);
	read_time = BILLION*(end_read.tv_sec-start_read.tv_sec) + (end_read.tv_nsec - start_read.tv_nsec);
	printf("Time taken to read the file in nano seconds is  %lu \n" , read_time);


	mcm_cuda_sha256_hash_batch(cpu_data, block_size, hash_cpu, total_blocks);


	total_dtrans_gpu_time = get_event_time(dtrans_to_gpu);
	total_dtrans_cpu_time = diff_events(kernel_launch, dtrans_to_cpu);
	kernel_execution_time = diff_events(dtrans_to_gpu,kernel_launch);
	total_time = total_dtrans_gpu_time + total_dtrans_cpu_time + kernel_execution_time;
	printf("Total Time to transfer data from CPU to GPU in milliseconds = %lf\n",total_dtrans_gpu_time);

	printf("Total Time to transfer data from GPU to CPU in milliseconds = %lf\n",total_dtrans_cpu_time);
	printf("Total kernel execution time in milliseconds = %lf\n",kernel_execution_time);
	
	total_gpu= BILLION*(end_gpu.tv_sec-start_gpu.tv_sec) + (end_gpu.tv_nsec - start_gpu.tv_nsec);

	printf("Total OPENCL Elapsed time in milliseconds =%lf with CPU clock=%lu\n",total_time,total_gpu);


	fclose(fptr);

//File output operations

	struct timespec start_write, end_write;
	uint64_t  write_time;
 		
	clock_gettime(CLOCK_MONOTONIC,&start_write);
	FILE *fptr_out = fopen(argv[4],"wb");
	fwrite(hash_cpu,sizeof(unsigned char),total_blocks*SHA256_BLOCK_SIZE,fptr_out);
/*	
	for (i=0;i<total_strings;i++) 
	{
		fread(&cpu_data[i][0],1,string_length,fptr);
		cpu_data[i][string_length]='$';
	} */
	clock_gettime(CLOCK_MONOTONIC,&end_write);
	write_time = BILLION*(end_write.tv_sec-start_write.tv_sec) + (end_write.tv_nsec - start_write.tv_nsec);
	printf("Time taken to write the file in nano seconds is  %lu \n" , write_time);



/*	
	int i,j;
	for(i=0;i<30;i++)
	{
		for(j=i*32;j<(i+1)*32;j++)
			printf("Hash Value %d -- %x",i,hash_cpu[j]); 
		printf("\n");
	}*/
	return 0;



}



