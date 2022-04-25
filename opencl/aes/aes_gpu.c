#include <stdio.h>
#include <stdint.h>
#include<stdlib.h>
#include<inttypes.h>
#include<string.h>
#include<assert.h>
#include<time.h>
#include<errno.h>
#include<math.h>
#include <CL/cl.h>

#define BILLION 1E9

#define MAX_SOURCE_SIZE (0x100000)
#define THREADS_PER_GROUP 256

#define AES_BLOCK_SIZE      16
#define AES_ROUNDS          10  // 12, 14
#define AES_ROUND_KEY_SIZE  176 // AES-128 has 10 rounds, and there is a AddRoundKey before first round. (10+1)x16=176.


//__constant__ uint8_t S_gpu[256];
//__constant__ uint8_t RC_gpu[10];

cl_assert(cl_int  const code , char const * const file, int const line) {

	if(code!=CL_SUCCESS) 
	{
		fprintf(stderr,"\"%s\", line %d, cl_assert (%d) \n",file,line,code);
	}

}

//Difference between start and end to two different events
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

/*
 * round constants
 */
static uint8_t RC[] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};

/*
 * Sbox
 */
uint8_t SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

/*
 * Inverse Sboxs
 */
static uint8_t INV_SBOX[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d};

void aes_key_schedule_128(const uint8_t *key, uint8_t *roundkeys) {

    uint8_t temp[4];
    uint8_t *last4bytes; // point to the last 4 bytes of one round
    uint8_t *lastround;
    uint8_t i;

    for (i = 0; i < 16; ++i) {
        *roundkeys++ = *key++;
    }

    last4bytes = roundkeys-4;
    for (i = 0; i < AES_ROUNDS; ++i) {
        // k0-k3 for next round
        temp[3] = SBOX[*last4bytes++];
        temp[0] = SBOX[*last4bytes++];
        temp[1] = SBOX[*last4bytes++];
        temp[2] = SBOX[*last4bytes++];
        temp[0] ^= RC[i];
        lastround = roundkeys-16;
        *roundkeys++ = temp[0] ^ *lastround++;
        *roundkeys++ = temp[1] ^ *lastround++;
        *roundkeys++ = temp[2] ^ *lastround++;
        *roundkeys++ = temp[3] ^ *lastround++;
        // k4-k7 for next round        
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        // k8-k11 for next round
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        // k12-k15 for next round
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
    }

}
/*
__device__ void aes_key_schedule_128_gpu(const uint8_t *key, uint8_t *roundkeys) {

    uint8_t temp[4];
    uint8_t *last4bytes; // point to the last 4 bytes of one round
    uint8_t *lastround;
    uint8_t i;

    for (i = 0; i < 16; ++i) {
        *roundkeys++ = *key++;
    }

    last4bytes = roundkeys-4;
    for (i = 0; i < AES_ROUNDS; ++i) {
        // k0-k3 for next round
        temp[3] = S_gpu[*last4bytes++];
        temp[0] = S_gpu[*last4bytes++];
        temp[1] = S_gpu[*last4bytes++];
        temp[2] = S_gpu[*last4bytes++];
        temp[0] ^= RC_gpu[i];
        lastround = roundkeys-16;
        *roundkeys++ = temp[0] ^ *lastround++;
        *roundkeys++ = temp[1] ^ *lastround++;
        *roundkeys++ = temp[2] ^ *lastround++;
        *roundkeys++ = temp[3] ^ *lastround++;
        // k4-k7 for next round        
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        // k8-k11 for next round
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        // k12-k15 for next round
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
        *roundkeys++ = *last4bytes++ ^ *lastround++;
    }
}
*/

int main(int argc, char *argv[]) {

	printf("Usage: ./aes_gpu <input_filename>  <total_blocks> <output_filename> \n");

	uint8_t i, r;

	/* 128 bit key */
	uint8_t key[] = {
		//0x0f, 0x15, 0x71, 0xc9, 0x47, 0xd9, 0xe8, 0x59, 
		//0x0c, 0xb7, 0xad, 0xd6, 0xaf, 0x7f, 0x67, 0x98,
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 

	};

 
	uint8_t ciphertext[AES_BLOCK_SIZE];

	uint8_t roundkeys[AES_ROUND_KEY_SIZE];
//	uint8_t *plaintext_gpu, *ciphertext_gpu, *roundkeys_gpu;
	uint8_t *plaintext_cpu, *ciphertext_cpu;

	struct timespec start_read, end_read;
	uint64_t  read_time;
	
	//cudaEvent_t start,stop;
	//cudaEvent_t stop_dtrans_gpu;  //To track CPU to GPU data transfer
	//udaEvent_t start_dtrans_cpu; //To track GPU to CPU data transfer

	double total_time;	
	double total_dtrans_gpu_time;
	double total_dtrans_cpu_time;
	double kernel_execution_time;

 
  	unsigned int total_blocks=0;
	unsigned int total_threads=0;
	unsigned int thread_blocks;
	unsigned int size;

// Load the kernel source code into the array source_str
        FILE *kernel_fp;
    	char *kernel_str;
	size_t kernel_str_size;
  	cl_event dtrans1_to_gpu,dtrans2_to_gpu;
	cl_event dtrans_to_cpu;
	cl_event kernel_launch;
  	
	
   	kernel_fp = fopen("aes_gpu.cl", "r");
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





	total_blocks = atoi(argv[2]);
	total_threads=total_blocks;
	
	size=AES_BLOCK_SIZE*total_blocks;

        printf("total_text_blocks = %u, size = %u\n",total_blocks,size);
	plaintext_cpu = (uint8_t*)malloc(size*sizeof(uint8_t));
	ciphertext_cpu = (uint8_t*)malloc(size*sizeof(uint8_t));
	
	clock_gettime(CLOCK_MONOTONIC,&start_read);
	FILE *fptr = fopen(argv[1],"rb");
	fread(plaintext_cpu,size,1,fptr);

	clock_gettime(CLOCK_MONOTONIC,&end_read);
	read_time = BILLION*(end_read.tv_sec-start_read.tv_sec) + (end_read.tv_nsec - start_read.tv_nsec);
	printf("Time taken to read the file in nano seconds is  %lu \n" , read_time);

	aes_key_schedule_128(key, roundkeys);
	struct timespec start_gpu, end_gpu;
	uint64_t  total_gpu;


	clock_gettime(CLOCK_MONOTONIC,&start_gpu);
	cl_mem plaintext_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, size*sizeof(uint8_t),NULL, &ret);
cl_ok(ret);
	cl_mem ciphertext_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size*sizeof(uint8_t),NULL, &ret);
cl_ok(ret);
	cl_mem roundkeys_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, AES_ROUND_KEY_SIZE*sizeof(uint8_t),NULL,&ret);
cl_ok(ret);
	cl(EnqueueWriteBuffer(command_queue,plaintext_gpu,CL_TRUE,0,size*sizeof(uint8_t),plaintext_cpu,0,NULL,&dtrans1_to_gpu));
	cl(EnqueueWriteBuffer(command_queue,roundkeys_gpu,CL_TRUE,0,AES_ROUND_KEY_SIZE*sizeof(uint8_t),roundkeys,0,NULL,&dtrans2_to_gpu));
	cl_program program = clCreateProgramWithSource(context,1,(const char **)&kernel_str, (const size_t *)&kernel_str_size, &ret);
cl_ok(ret);
	ret=clBuildProgram(program, 1, &device_id,NULL,NULL,NULL);
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
	cl_kernel kernel = clCreateKernel(program,"aes_encrypt_gpu",&ret);
cl_ok(ret);
	cl(SetKernelArg(kernel,0,sizeof(cl_mem),(void*)&plaintext_gpu));
	cl(SetKernelArg(kernel,1,sizeof(cl_mem),(void*)&ciphertext_gpu));
	cl(SetKernelArg(kernel,2,sizeof(cl_mem),(void*)&roundkeys_gpu));
	cl(SetKernelArg(kernel,3,sizeof(unsigned int),(void*)&total_blocks));
size_t local_item_size = THREADS_PER_GROUP;
size_t global_item_size = ((size_t)ceil((double)(total_blocks)/(double)(THREADS_PER_GROUP)))*THREADS_PER_GROUP;
	cl(EnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0,NULL,&kernel_launch));
	cl(EnqueueReadBuffer(command_queue,ciphertext_gpu, CL_TRUE,0, size*sizeof(uint8_t),ciphertext_cpu,0,NULL,&dtrans_to_cpu));	
	total_dtrans_gpu_time = diff_events_start_end(dtrans1_to_gpu,dtrans2_to_gpu);
	total_dtrans_cpu_time = diff_events(kernel_launch, dtrans_to_cpu);
	kernel_execution_time = diff_events(dtrans2_to_gpu,kernel_launch);
	total_time = total_dtrans_gpu_time + total_dtrans_cpu_time + kernel_execution_time;
	

	printf("Total Time to transfer data from CPU to GPU in milliseconds = %lf\n",total_dtrans_gpu_time);

	printf("Total Time to transfer data from GPU to CPU in milliseconds = %lf\n",total_dtrans_cpu_time);
//	kernel_execution_time = total_time - (total_dtrans_gpu_time+total_dtrans_cpu_time);
	printf("Total kernel execution time in milliseconds = %lf\n",kernel_execution_time);
	
	cl(Flush(command_queue));
	cl(Finish(command_queue));

	clock_gettime(CLOCK_MONOTONIC,&end_gpu);
	total_gpu= BILLION*(end_gpu.tv_sec-start_gpu.tv_sec) + (end_gpu.tv_nsec - start_gpu.tv_nsec);
	printf("Total OPENCL Elapsed time in milliseconds =%lf with CPU clock=%lu\n",total_time,total_gpu);
	//printf("GPU time measured in nano seconds from CPU clock  %lu \n" , total_gpu);



	cl(ReleaseKernel(kernel));
	cl(ReleaseProgram(program));
	cl(ReleaseMemObject(plaintext_gpu));
	cl(ReleaseMemObject(ciphertext_gpu));
	cl(ReleaseMemObject(roundkeys_gpu));
	cl(ReleaseCommandQueue(command_queue));
	cl(ReleaseContext(context));
	fclose(fptr);

//        thread_blocks=(total_threads/THREADS_PER_BLOCK) + 1;

//	cudaEventCreate(&start);
//      cudaEventCreate(&stop);
//    	cudaEventCreate(&start_dtrans_cpu);
//	cudaEventCreate(&stop_dtrans_gpu);

//	cudaEventRecord(start, 0);
//      cudaMalloc((void**)&plaintext_gpu,size*sizeof(uint8_t));
//      cudaMalloc((void**)&ciphertext_gpu,size*sizeof(uint8_t));
//      cudaMalloc((void**)&roundkeys_gpu,AES_ROUND_KEY_SIZE*sizeof(uint8_t));


//	cudaErrorCheck(cudaMemcpyToSymbol(S_gpu,SBOX,256*sizeof(uint8_t)));
//	cudaErrorCheck(cudaMemcpyToSymbol(RC_gpu,RC,10*sizeof(uint8_t)));

 //       cudaErrorCheck(cudaMemcpy(plaintext_gpu,plaintext_cpu,size*sizeof(uint8_t),cudaMemcpyHostToDevice));
//        cudaErrorCheck(cudaMemcpy(roundkeys_gpu,roundkeys,AES_ROUND_KEY_SIZE*sizeof(uint8_t),cudaMemcpyHostToDevice));

//	cudaEventRecord(stop_dtrans_gpu,0);

//	aes_encrypt_gpu<<<thread_blocks,THREADS_PER_BLOCK>>>(plaintext_gpu,ciphertext_gpu,roundkeys_gpu,total_blocks) ;

//      cudaErrorCheck(cudaPeekAtLastError());
	
//	cudaEventRecord(start_dtrans_cpu,0);
//      cudaErrorCheck(cudaMemcpy(ciphertext_cpu,ciphertext_gpu,size*sizeof(uint8_t),cudaMemcpyDeviceToHost));

//      cudaEventRecord(stop, 0);
//      cudaEventSynchronize(stop);
//     	cudaEventElapsedTime(&total_time, start, stop);
//	cudaEventElapsedTime(&total_dtrans_gpu_time,start,stop_dtrans_gpu);
//	cudaEventElapsedTime(&total_dtrans_cpu_time,start_dtrans_cpu,stop);
//	printf("Total Time to transfer data from CPU to GPU in milliseconds = %lf\n",total_dtrans_gpu_time);

//	printf("Total Time to transfer data from GPU to CPU in milliseconds = %lf\n",total_dtrans_cpu_time);
//	kernel_execution_time = total_time - (total_dtrans_gpu_time+total_dtrans_cpu_time);
//	printf("Total kernel execution time in milliseconds = %lf\n",kernel_execution_time);
//	printf("Total CUDA Elapsed time in millisecond=%lf\n",total_time);
//        cudaEventDestroy(start);
//        cudaEventDestroy(stop);
//	cudaEventDestroy(start_dtrans_cpu);
//	cudaEventDestroy(stop_dtrans_gpu);
//	cudaFree(ciphertext_gpu);
//	cudaFree(plaintext_gpu);
//	cudaFree(roundkeys_gpu);
//	fclose(fptr);
	//File output operations

	struct timespec start_write, end_write;
	uint64_t  write_time;
 		
	clock_gettime(CLOCK_MONOTONIC,&start_write);
	FILE *fptr_out = fopen(argv[3],"wb");
	fwrite(ciphertext_cpu,size,1,fptr_out);
/*	
	for (i=0;i<total_strings;i++) 
	{
		fread(&cpu_data[i][0],1,string_length,fptr);
		cpu_data[i][string_length]='$';
	} */
	clock_gettime(CLOCK_MONOTONIC,&end_write);
	write_time = BILLION*(end_write.tv_sec-start_write.tv_sec) + (end_write.tv_nsec - start_write.tv_nsec);
	printf("Time taken to write the file in nano seconds is  %lu \n" , write_time);


	fclose(fptr_out);
	return 0;

}
