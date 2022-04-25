#include <cuda.h>
#include<stdint.h>
#include<stdio.h>
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


__global__  void gpu_fast_page_hash(char *page_data, uint32_t *page_hash_gpu,int page_size,int total_threads) {

	int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	uint32_t len = page_size;
	uint32_t hash ,tmp;
	char *temp_ptr;
	hash=len;
	int rem;

	int thread_page_index=thread_id*page_size;
	//	page_hashes->physical_id=data->physical_id;
	temp_ptr=page_data+thread_page_index;
	//	temp_ptr=data->data[thread_id];

	//	int test=0;
	//	if(test==1) {	
	if(thread_id<total_threads) {
		//if (len <= 0 || data == NULL) return 0;
		rem = len & 3; // And operation netween PAGE_SIZE and 11 (which is 3)
		len >>= 2; // left shift page size. 4096>>2 = 1024 
		/* Main loop */
		for (;len > 0; len--) {  //will run for 1024 times. 
			hash  += get16bits (temp_ptr);  //Sum first two bytes of data char array
			tmp    = (get16bits (temp_ptr+2) << 11) ^ hash;  //sum of 3rd and 4th byte of data char array xored with sum of 1st two bytes
			hash   = (hash << 16) ^ tmp;
			temp_ptr  += 2*sizeof (uint16_t);
			hash  += hash >> 11;
		}

		/* Handle end cases */

		switch (rem) {
		case 3: hash += get16bits (temp_ptr);
		hash ^= hash << 16;
		hash ^= ((signed char)temp_ptr[sizeof (uint16_t)]) << 18;
		hash += hash >> 11;
		break;
		case 2: hash += get16bits (temp_ptr);
		hash ^= hash << 11;
		hash += hash >> 17;
		break;
		case 1: hash += (signed char)*temp_ptr;
		hash ^= hash << 10;
		hash += hash >> 1;
		}		

		/* Force "avalanching" of final 127 bits */

		hash ^= hash << 3;
		hash += hash >> 5;
		hash ^= hash << 4;
		hash += hash >> 17;
		hash ^= hash << 25;
		hash += hash >> 6;
		//	page_hashes->hashes[thread_id]=hash;
		//	page_hashes->virtual_address[thread_id]=data->virtual_address[thread_id];
		//	page_hash[thread_id]=hash;
		page_hash_gpu[thread_id]=hash;
		//return hash; Instead of returning store this value in some array
		//	page_hash[thread_id]=thread_id;
	}
}



int main(int argc, char *argv[]) {


	printf("Usage:  ./fast_hash <input_filename> <total_blocks> <block_size>  <outfile_name> \n");

	unsigned int block_size, total_blocks;
	char *cpu_data;
	char *gpu_data;


	uint32_t *hash_gpu;
	uint32_t *hash_cpu;

	unsigned int total_threads =0;
	unsigned int thread_blocks =0;
	unsigned int size;

	cudaEvent_t start,stop;
	cudaEvent_t stop_dtrans_gpu;  //To track CPU to GPU data transfer
	cudaEvent_t start_dtrans_cpu; //To track GPU to CPU data transfer
	float total_time;	
	float total_dtrans_gpu_time;
	float total_dtrans_cpu_time;
	float kernel_execution_time;


	struct timespec start_read, end_read;
	uint64_t  read_time;
	


	total_blocks=atoi(argv[2]);
	block_size = atoi(argv[3]); 
	total_threads = total_blocks;
        size= total_threads*block_size;

	cpu_data = (char*)malloc(size*sizeof(char));
	hash_cpu = (uint32_t *)malloc(total_blocks*sizeof(uint32_t));


	clock_gettime(CLOCK_MONOTONIC,&start_read);
	FILE *fptr = fopen(argv[1],"rb");
	fread(cpu_data,size,1,fptr);
	clock_gettime(CLOCK_MONOTONIC,&end_read);
	read_time = BILLION*(end_read.tv_sec-start_read.tv_sec) + (end_read.tv_nsec - start_read.tv_nsec);
	printf("Time taken to read the file in nano seconds is  %lu \n" , read_time);

	thread_blocks=(total_threads/THREADS_PER_BLOCK) + 1;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start_dtrans_cpu);
	cudaEventCreate(&stop_dtrans_gpu);

	cudaEventRecord(start, 0);

	cudaMalloc((void**)&gpu_data,size*sizeof(char));
	cudaMalloc((void**)&hash_gpu,total_blocks*sizeof(uint32_t));
	cudaErrorCheck(cudaMemcpy(gpu_data,cpu_data,size,cudaMemcpyHostToDevice));

	cudaEventRecord(stop_dtrans_gpu,0);

	gpu_fast_page_hash<<<thread_blocks,THREADS_PER_BLOCK>>>(gpu_data,hash_gpu,block_size,total_threads);

	cudaEventRecord(start_dtrans_cpu,0);
//	cudaEventSynchronize(start_dtrans_cpu);
	cudaErrorCheck(cudaMemcpy(hash_cpu,hash_gpu,total_blocks*sizeof(uint32_t),cudaMemcpyDeviceToHost));

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
	fclose(fptr);
	//File output operations

	struct timespec start_write, end_write;
	uint64_t  write_time;
 		
	clock_gettime(CLOCK_MONOTONIC,&start_write);
	FILE *fptr_out = fopen(argv[4],"wb");
	fwrite(hash_cpu,sizeof(uint32_t),total_blocks,fptr_out);
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
	int i;
	for(i=0;i<total_blocks;i++)
		printf("Hash Value %d -- %u \n",i,hash_cpu[i]); */
	return 0;

}



