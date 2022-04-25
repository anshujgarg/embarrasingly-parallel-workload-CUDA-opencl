#include<stdint.h>
#include<stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#define BILLION 1000000000L

#define MAX_SOURCE_SIZE (0x100000)
#define THREADS_PER_GROUP 64

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


int main(int argc, char *argv[]) {

	printf("Usage: ./psearch  <input_list>  <query_list> <list_size>  <num_queries> <outfile> \n");
	//cudaEvent_t start,stop;
	//cudaEvent_t stop_dtrans_gpu;  //To track CPU to GPU data transfer
	//cudaEvent_t start_dtrans_cpu; //To track GPU to CPU data transfer

	double total_time;	
	double total_dtrans_gpu_time;
	double total_dtrans_cpu_time;
	double kernel_execution_time;


	struct timespec start_read, end_read;
	uint64_t  read_time;
	
	unsigned int list_size;
	unsigned int total_queries;
	
	unsigned int thread_blocks;
	unsigned int total_threads;
	uint8_t *present_cpu;
	unsigned int *cpu_list, *cpu_query;

// Load the kernel source code into the array source_str
        FILE *kernel_fp;
    	char *kernel_str;
	size_t kernel_str_size;
	cl_event dtrans1_to_gpu,dtrans2_to_gpu;
	cl_event dtrans_to_cpu;
	cl_event kernel_launch;
  	
   	kernel_fp = fopen("psearch.cl", "r");
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



	list_size=atoi(argv[3]);
	total_queries=atoi(argv[4]);

//	unsigned int i,j;
	cpu_list = (unsigned int*)malloc(list_size*sizeof(unsigned int));
	cpu_query = (unsigned int *)malloc(total_queries*sizeof(unsigned int));
	present_cpu= (uint8_t *)malloc(total_queries*sizeof(uint8_t));
	printf("list size =%u, total_queries=%u\n",list_size, total_queries);
	clock_gettime(CLOCK_MONOTONIC,&start_read);
	FILE *fptr = fopen(argv[1],"rb");
	fread(cpu_list,sizeof(unsigned int),list_size,fptr);
	clock_gettime(CLOCK_MONOTONIC,&end_read);
	read_time = BILLION*(end_read.tv_sec-start_read.tv_sec) + (end_read.tv_nsec - start_read.tv_nsec);
	printf("Time taken to read the source list file in nano seconds is  %lu \n" , read_time);
	
	fclose(fptr);

	clock_gettime(CLOCK_MONOTONIC,&start_read);
	fptr = fopen(argv[2],"rb");
	fread(cpu_query,sizeof(unsigned int),total_queries,fptr);
	clock_gettime(CLOCK_MONOTONIC,&end_read);
	read_time = BILLION*(end_read.tv_sec-start_read.tv_sec) + (end_read.tv_nsec - start_read.tv_nsec);
	printf("Time taken to read the query file in nano seconds is  %lu \n" , read_time);
	
	fclose(fptr);


	total_threads= total_queries;
/*
	unsigned int temp;
	printf("rand %u \n",rand()); 
	for (i=0;i<list_size;i++)
	{
		temp = rand();
		cpu_list[i] = temp;
	}
	for (i=0;i<total_queries;i++)
	{
		temp= rand();
		cpu_query[i] = temp;
	}
	*/
/*
	clock_gettime(CLOCK_MONOTONIC,&start_read);
	FILE *fptr = fopen(argv[1],"rb");
	fread(cpu_data,size,1,fptr);
	clock_gettime(CLOCK_MONOTONIC,&end_read);
	read_time = BILLION*(end_read.tv_sec-start_read.tv_sec) + (end_read.tv_nsec - start_read.tv_nsec);
	printf("Time taken to read the file in nano seconds is  %lu \n" , read_time);
*/
//	thread_blocks=(total_threads/THREADS_PER_BLOCK) + 1;


	struct timespec start_gpu, end_gpu;
	uint64_t  total_gpu;


	clock_gettime(CLOCK_MONOTONIC,&start_gpu);

	cl_mem gpu_list = clCreateBuffer(context, CL_MEM_READ_ONLY, list_size*sizeof(unsigned int),NULL, &ret);
	cl_ok(ret);

	cl_mem gpu_query = clCreateBuffer(context, CL_MEM_READ_ONLY, total_queries*sizeof(unsigned int),NULL, &ret);
	cl_ok(ret);
	
	cl_mem present_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_queries*sizeof(unsigned char ),NULL,&ret);
	cl_ok(ret);

	cl(EnqueueWriteBuffer(command_queue,gpu_list,CL_TRUE,0,list_size*sizeof(unsigned int),cpu_list,0,NULL,&dtrans1_to_gpu));
	cl(EnqueueWriteBuffer(command_queue,gpu_query,CL_TRUE,0,total_queries*sizeof(unsigned int),cpu_query,0,NULL,&dtrans2_to_gpu));
	

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
	cl_kernel kernel = clCreateKernel(program,"gpu_psearch",&ret);
cl_ok(ret);
	cl(SetKernelArg(kernel,0,sizeof(cl_mem),(void*)&gpu_list));
	cl(SetKernelArg(kernel,1,sizeof(cl_mem),(void*)&gpu_query));
	cl(SetKernelArg(kernel,2,sizeof(cl_mem),(void*)&present_gpu));
	cl(SetKernelArg(kernel,3,sizeof(unsigned int),(void*)&total_queries));
	cl(SetKernelArg(kernel,4,sizeof(unsigned int),(void*)&list_size));
size_t local_item_size = THREADS_PER_GROUP;
size_t global_item_size = ((size_t)ceil((double)(total_threads)/(double)(THREADS_PER_GROUP)))*THREADS_PER_GROUP;
	cl(EnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0,NULL,&kernel_launch));

	cl(EnqueueReadBuffer(command_queue,present_gpu, CL_TRUE,0, total_queries*sizeof(unsigned char),present_cpu,0,NULL,&dtrans_to_cpu));	
	total_dtrans_gpu_time = diff_events_start_end(dtrans1_to_gpu,dtrans2_to_gpu);
	total_dtrans_cpu_time = diff_events(kernel_launch, dtrans_to_cpu);
	kernel_execution_time = diff_events(dtrans2_to_gpu,kernel_launch);
	total_time = total_dtrans_gpu_time + total_dtrans_cpu_time + kernel_execution_time;
	printf("Total Time to transfer data from CPU to GPU in milliseconds = %lf\n",total_dtrans_gpu_time);

	printf("Total Time to transfer data from GPU to CPU in milliseconds = %lf\n",total_dtrans_cpu_time);
	printf("Total kernel execution time in milliseconds = %lf\n",kernel_execution_time);
	
	cl(Flush(command_queue));
	cl(Finish(command_queue));

	clock_gettime(CLOCK_MONOTONIC,&end_gpu);
	total_gpu= BILLION*(end_gpu.tv_sec-start_gpu.tv_sec) + (end_gpu.tv_nsec - start_gpu.tv_nsec);
	printf("Total OPENCL Elapsed time in milliseconds =%lf with CPU clock=%lu\n",total_time,total_gpu);



	cl(ReleaseKernel(kernel));
	cl(ReleaseProgram(program));
	cl(ReleaseMemObject(gpu_list));
	cl(ReleaseMemObject(gpu_query));
	cl(ReleaseMemObject(present_gpu));
	cl(ReleaseCommandQueue(command_queue));
	cl(ReleaseContext(context));
//	fclose(fptr);
	//File output operations
/*	
	for (i=0;i<total_queries; i++)
		printf("%u -- %u \t",i,present_cpu[i]);
*/
	struct timespec start_write, end_write;
	uint64_t  write_time;
 		
	clock_gettime(CLOCK_MONOTONIC,&start_write);
	FILE *fptr_out = fopen(argv[5],"wb");
	fwrite(present_cpu,sizeof(uint8_t),total_queries,fptr_out);
/*	
	for (i=0;i<total_strings;i++) 
	{
		fread(&cpu_data[i][0],1,string_length,fptr);
		cpu_data[i][string_length]='$';
	} */
	clock_gettime(CLOCK_MONOTONIC,&end_write);
	write_time = BILLION*(end_write.tv_sec-start_write.tv_sec) + (end_write.tv_nsec - start_write.tv_nsec);
	printf("Time taken to write the file in nano seconds is  %lu \n" , write_time);



return 0;
}


