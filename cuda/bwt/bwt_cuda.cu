// C program to find Burrows Wheeler transform 
// of a given text 

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <stdint.h>
#include <cuda.h>

#define BILLION 1000000000L

#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }


#define THREADS_PER_BLOCK 256

inline void cudaAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)  {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}







// Structure to store data of a rotation 
struct rotation { 
	int index; 
	char* suffix; 
}; 


 

/* you can also write compare function for floats, 
	chars, double similarly as integer. */
// function for swap two elements 
__device__ void swap_gpu(void* v1, void* v2, int size) 
{ 
	// buffer is array of characters which will 
	// store element byte by byte 
	//char buffer[size]; 
	char *buffer = (char*)malloc(size*sizeof(char));

	// memcpy will copy the contents from starting 
	// address of v1 to length of size in buffer 
	// byte by byte. 
	memcpy(buffer, v1, size); 
	memcpy(v1, v2, size); 
	memcpy(v2, buffer, size); 
	free(buffer);
} 



// v is an array of elements to sort. 
// size is the number of elements in array 
// left and right is start and end of array 
//(*comp)(void*, void*) is a pointer to a function 
// which accepts two void* as its parameter 
__device__ void _qsort_gpu(void* v, int size, int left, int right, 
					int (*comp)(void*, void*)) 
{ 
	void *vt, *v3; 
	int i, last, mid = (left + right) / 2; 
	if (left >= right) 
		return; 

	// casting void* to char* so that operations 
	// can be done. 
	void* vl = (char*)(v + (left * size)); 
	void* vr = (char*)(v + (mid * size)); 
	swap_gpu(vl, vr, size); 
	last = left; 
	for (i = left + 1; i <= right; i++) { 

		// vl and vt will have the starting address 
		// of the elements which will be passed to 
		// comp function. 
		vt = (char*)(v + (i * size)); 
		if ((*comp)(vl, vt) > 0) { 
			++last; 
			v3 = (char*)(v + (last * size)); 
			swap_gpu(vt, v3, size); 
		} 
	} 
	v3 = (char*)(v + (last * size)); 
	swap_gpu(vl, v3, size); 
	_qsort_gpu(v, size, left, last - 1, comp); 
	_qsort_gpu(v, size, last + 1, right, comp); 
} 

__device__ int strcmp_gpu (const char * s1, const char * s2) {
	for(; *s1 == *s2; ++s1, ++s2)
        	if(*s1 == 0)
	                return 0;
		return *(unsigned char *)s1 < *(unsigned char *)s2 ? -1 : 1;
		}




// Compares the rotations and 
// sorts the rotations alphabetically 
__device__ int cmpfunc_gpu(void* x, void* y) 
{ 
	struct rotation* rx = (struct rotation*)x; 
	struct rotation* ry = (struct rotation*)y; 
	return strcmp_gpu(rx->suffix, ry->suffix); 
}




// Takes suffix array and its size 
// as arguments and returns the 
// Burrows - Wheeler Transform of given text 
__device__ char* findLastChar_gpu(char* input_text, 
				int* suffix_arr, int n) 
{ 
	// Iterates over the suffix array to find 
	// the last char of each cyclic rotation 
	char* bwt_arr = (char*)malloc(n * sizeof(char)); 
	int i; 
	for (i = 0; i < n; i++) { 
		// Computes the last char which is given by 
		// input_text[(suffix_arr[i] + n - 1) % n] 
		int j = suffix_arr[i] - 1; 
		if (j < 0) 
			j = j + n; 

		bwt_arr[i] = input_text[j]; 
	} 

//	bwt_arr[i] = '\0'; 

	// Returns the computed Burrows - Wheeler Transform 
	return bwt_arr; 
} 



__device__ int* computeSuffixArray_gpu(char* input_text, int len_text) 
{ 
	// Array of structures to store rotations and 
	// their indexes 
	//struct rotation suff[len_text];
	
	
	int i=0;

	
	struct rotation *suff =(struct rotation*)malloc(len_text*sizeof(struct rotation));
	if(suff==NULL)
		printf("Got null value\n");


	// Structure is needed to maintain old indexes of 
	// rotations after sorting them 

	for (i = 0; i < len_text; i++) { 
		suff[i].index = i; 
		suff[i].suffix = (input_text + i); 
	} 


	_qsort_gpu(suff,sizeof(struct rotation) ,0,len_text-1, cmpfunc_gpu); 

	int* suffix_arr 
		= (int*)malloc(len_text * sizeof(int)); 
	for (i = 0; i < len_text; i++) 
//		suffix_arr[i]=1;
		suffix_arr[i] = suff[i].index; 
	free(suff);

	// Returns the computed suffix array 
	return suffix_arr; 
} 





//GPU kernel to compute BWT




__global__ void bwt_gpu_func(char *strings, char *bwt, unsigned int string_length, unsigned int total_strings) {

	unsigned int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int offset = thread_id*string_length;
	unsigned int end_offset = (thread_id+1)*string_length;
	int *suffix_arr;
	char * bwt_arr;
	int i,j;
	if(thread_id < total_strings) {

		suffix_arr = computeSuffixArray_gpu(&strings[offset],string_length);	
		bwt_arr = findLastChar_gpu(&strings[offset],suffix_arr,string_length);
		free(suffix_arr);
		for(i=offset,j=0;i<end_offset;i++,j++) {
			bwt[i]=bwt_arr[j];
//			bwt[i]='A';
		}
		free(bwt_arr);

	}


		
}




// Driver program to test functions above 
int main(int argc, char *argv[]) 
{



	printf("Usage:  ./bwt_cuda <input_filename> <total_strings> <string_length> <output_filename>\n");

	
	unsigned int string_length, total_strings;
	char *cpu_data;
	char *gpu_data;


	char *bwt_gpu;
	char *bwt_cpu;

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
	

	int i,j;

	total_strings=atoi(argv[2]);
	string_length = atoi(argv[3]); 
	total_threads = total_strings;
        size= total_threads*string_length;

	cpu_data = (char*)malloc(size*sizeof(char));
	bwt_cpu = (char *)malloc(size*sizeof(char));

/*	for (i=0;i<total_strings;i++) {
		cpu_data[i]= (char*)malloc((string_length+1)*sizeof(char));
		bwt_cpu[i]= (char*)malloc((string_length+1)*sizeof(char));
	}*/


	clock_gettime(CLOCK_MONOTONIC,&start_read);
	FILE *fptr = fopen(argv[1],"rb");
	fread(cpu_data,size,1,fptr);
/*	
	for (i=0;i<total_strings;i++) 
	{
		fread(&cpu_data[i][0],1,string_length,fptr);
		cpu_data[i][string_length]='$';
	} */
	clock_gettime(CLOCK_MONOTONIC,&end_read);
	read_time = BILLION*(end_read.tv_sec-start_read.tv_sec) + (end_read.tv_nsec - start_read.tv_nsec);
	printf("Time taken to read the file in nano seconds is  %lu \n" , read_time);

/*	for (i=0;i<total_strings;i++) {
		printf("string %d %s\n",i,cpu_data[i]);
	
	}*/


	size_t value;
	cudaThreadGetLimit(&value, cudaLimitMallocHeapSize);
	cudaThreadSetLimit(cudaLimitMallocHeapSize,128*1024*1024);
	printf("Heap limit is %d \n",value);
	thread_blocks=(total_threads/THREADS_PER_BLOCK) + 1;

	printf("Thread blocks =%u, total_threads = %u, total_strings = %u, string_ length = %u\n",thread_blocks,total_threads, total_strings, string_length);


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start_dtrans_cpu);
	cudaEventCreate(&stop_dtrans_gpu);
	
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&gpu_data,size*sizeof(char));
	cudaMalloc((void**)&bwt_gpu,size*sizeof(char));
	cudaErrorCheck(cudaMemcpy(gpu_data,cpu_data,size*sizeof(char),cudaMemcpyHostToDevice));

	cudaEventRecord(stop_dtrans_gpu,0);

	bwt_gpu_func<<<thread_blocks,THREADS_PER_BLOCK>>>(gpu_data,bwt_gpu,string_length,total_strings);
	cudaErrorCheck(cudaPeekAtLastError());

	cudaEventRecord(start_dtrans_cpu,0);

	cudaErrorCheck(cudaMemcpy(bwt_cpu,bwt_gpu,size*sizeof(char),cudaMemcpyDeviceToHost));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&total_time, start, stop);
	cudaEventElapsedTime(&total_dtrans_gpu_time,start,stop_dtrans_gpu);
	cudaEventElapsedTime(&total_dtrans_cpu_time,start_dtrans_cpu,stop);
	printf("Total Time to transfer data from CPU to GPU in milliseconds = %lf\n",total_dtrans_gpu_time);

	printf("Total Time to transfer data from GPU to CPU in milliseconds = %lf\n",total_dtrans_cpu_time);
	kernel_execution_time = total_time - (total_dtrans_gpu_time+total_dtrans_cpu_time);
	printf("Total kernel execution time in milliseconds = %lf\n",kernel_execution_time);
	printf("Total CUDA Elapsed time in millisecond=%lf\n",total_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(start_dtrans_cpu);
	cudaEventDestroy(stop_dtrans_gpu);
	cudaFree(bwt_gpu);
	cudaFree(gpu_data);
	fclose(fptr);

	struct timespec start_write, end_write;
	uint64_t  write_time;
 		
	clock_gettime(CLOCK_MONOTONIC,&start_write);
	FILE *fptr_out = fopen(argv[4],"wb");
	fwrite(bwt_cpu,size,1,fptr_out);
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
	for(i=0;i<100;i++)
	{
		if(i%string_length==0)
			printf("\n");

		printf("%c",bwt_cpu[i]); 
	}
*/

	return 0; 
} 

