#include <cuda.h>
#include<stdint.h>
#include<stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include <time.h>
#define BILLION 1000000000L

#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)+(uint32_t)(((const uint8_t *)(d))[0]))
#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }


#define THREADS_PER_BLOCK 512

inline void cudaAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)  {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}


__global__  void gpu_psearch(unsigned int *list, unsigned int *query,uint8_t *present, unsigned int total_queries, unsigned int list_size) {

	unsigned int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int i;
	if(thread_id < total_queries) {
		present[thread_id]=0;
		for (i=0;i<list_size;i++) {
			if(query[thread_id]==list[i])
				present[thread_id]=1;
		}
			

	}	
	
}


int main(int argc, char *argv[]) {

	printf("Usage: ./psearch  <input_list>  <query_list> <list_size>  <num_queries>  \n");
	cudaEvent_t start,stop;
	cudaEvent_t stop_dtrans_gpu;  //To track CPU to GPU data transfer
	cudaEvent_t start_dtrans_cpu; //To track GPU to CPU data transfer
	float total_time;	
	float total_dtrans_gpu_time;
	float total_dtrans_cpu_time;
	float kernel_execution_time;


	struct timespec start_read, end_read;
	uint64_t  read_time;
	
	unsigned int list_size;
	unsigned int total_queries;
	
	unsigned int thread_blocks;
	unsigned int total_threads;
	uint8_t *present_cpu;
	unsigned int *cpu_list, *cpu_query;
	unsigned int *gpu_list, *gpu_query;
	uint8_t *present_gpu;
//	total_blocks=atoi(argv[2]);
//	block_size = atoi(argv[3]); 
//	total_threads = total_blocks;
//        size= total_threads*block_size;


	list_size=atoi(argv[3]);
	total_queries=atoi(argv[4]);
//	list_size = atoi(argv[1]);
//	total_queries = atoi(argv[2]);

//	unsigned int i,j;
	cpu_list = (unsigned int*)malloc(list_size*sizeof(unsigned int));
	cpu_query = (unsigned int *)malloc(total_queries*sizeof(unsigned int));
	present_cpu= (uint8_t *)malloc(total_queries*sizeof(uint8_t));

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
	thread_blocks=(total_threads/THREADS_PER_BLOCK) + 1;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start_dtrans_cpu);
	cudaEventCreate(&stop_dtrans_gpu);
	
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&gpu_list,list_size*sizeof(unsigned int));
	cudaMalloc((void**)&gpu_query,total_queries*sizeof(unsigned int));
	cudaMalloc((void**)&present_gpu,total_queries*sizeof(uint8_t));

	cudaErrorCheck(cudaMemcpy(gpu_list,cpu_list,list_size*sizeof(unsigned int),cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(gpu_query,cpu_query,total_queries*sizeof(unsigned int),cudaMemcpyHostToDevice));

	cudaEventRecord(stop_dtrans_gpu,0);

	gpu_psearch<<<thread_blocks,THREADS_PER_BLOCK>>>(gpu_list,gpu_query,present_gpu,total_queries, list_size);

	cudaEventRecord(start_dtrans_cpu,0);
	cudaErrorCheck(cudaMemcpy(present_cpu,present_gpu,total_queries*sizeof(uint8_t),cudaMemcpyDeviceToHost));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&total_time, start, stop);
	cudaEventElapsedTime(&total_dtrans_gpu_time,start,stop_dtrans_gpu);
	cudaEventElapsedTime(&total_dtrans_cpu_time,start_dtrans_cpu,stop);
	printf("Total Time to transfer data from CPU to GPU in milliseconds = %lf\n",total_dtrans_gpu_time);

	printf("Total Time to transfer data from GPU to CPU in milliseconds = %lf\n",total_dtrans_cpu_time);
	kernel_execution_time = total_time - (total_dtrans_gpu_time+total_dtrans_cpu_time);
	printf("Total kernel execution time in milliseconds = %lf\n",kernel_execution_time);
	printf("Total CUDA Elapsed time in milliseconds =%lf\n",total_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(start_dtrans_cpu);
	cudaEventDestroy(stop_dtrans_gpu);
	cudaFree(present_gpu);
	cudaFree(gpu_query);
	cudaFree(gpu_list);

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


