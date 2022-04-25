#include <stdio.h>
#include <stdint.h>
#include<stdlib.h>
#include<inttypes.h>
#include<string.h>
#include<assert.h>
#include<time.h>
#include<errno.h>
#include<cuda.h>
#include<math.h>

#define BILLION 1E9
#define THREADS_PER_BLOCK 512 


#define AES_BLOCK_SIZE      16
#define AES_ROUNDS          10  // 12, 14
#define AES_ROUND_KEY_SIZE  176 // AES-128 has 10 rounds, and there is a AddRoundKey before first round. (10+1)x16=176.

inline __host__ void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
 if (code != cudaSuccess) 
 {
   fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code),file, line);
   if (abort) exit(code);
 }
}

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }



__constant__ uint8_t S_gpu[256];
__constant__ uint8_t RC_gpu[10];

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

/**
 * https://en.wikipedia.org/wiki/Finite_field_arithmetic
 * Multiply two numbers in the GF(2^8) finite field defined
 * by the polynomial x^8 + x^4 + x^3 + x + 1 = 0
 * We do use mul2(int8_t a) but not mul(uint8_t a, uint8_t b)
 * just in order to get a higher speed.
 */
__device__ static inline uint8_t mul2(uint8_t a) {
    return (a&0x80) ? ((a<<1)^0x1b) : (a<<1);
}

/**
 * @purpose:    ShiftRows
 * @descrption:
 *  Row0: s0  s4  s8  s12   <<< 0 byte
 *  Row1: s1  s5  s9  s13   <<< 1 byte
 *  Row2: s2  s6  s10 s14   <<< 2 bytes
 *  Row3: s3  s7  s11 s15   <<< 3 bytes
 */
__device__ static void shift_rows(uint8_t *state) {
    uint8_t temp;
    // row1
    temp        = *(state+1);
    *(state+1)  = *(state+5);
    *(state+5)  = *(state+9);
    *(state+9)  = *(state+13);
    *(state+13) = temp;
    // row2
    temp        = *(state+2);
    *(state+2)  = *(state+10);
    *(state+10) = temp;
    temp        = *(state+6);
    *(state+6)  = *(state+14);
    *(state+14) = temp;
    // row3
    temp        = *(state+15);
    *(state+15) = *(state+11);
    *(state+11) = *(state+7);
    *(state+7)  = *(state+3);
    *(state+3)  = temp;
}

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


__device__ void aes_encrypt_128_gpu(const uint8_t *roundkeys, const uint8_t *plaintext, uint8_t *ciphertext) {

    uint8_t tmp[16], t;
    uint8_t i, j;

    // first AddRoundKey
    for ( i = 0; i < AES_BLOCK_SIZE; ++i ) {
        *(ciphertext+i) = *(plaintext+i) ^ *roundkeys++;
    }

    // 9 rounds
    for (j = 1; j < AES_ROUNDS; ++j) {

        // SubBytes
        for (i = 0; i < AES_BLOCK_SIZE; ++i) {
            *(tmp+i) = S_gpu[*(ciphertext+i)];
        }
        shift_rows(tmp);
        /*
         * MixColumns 
         * [02 03 01 01]   [s0  s4  s8  s12]
         * [01 02 03 01] . [s1  s5  s9  s13]
         * [01 01 02 03]   [s2  s6  s10 s14]
         * [03 01 01 02]   [s3  s7  s11 s15]
         */
        for (i = 0; i < AES_BLOCK_SIZE; i+=4)  {
            t = tmp[i] ^ tmp[i+1] ^ tmp[i+2] ^ tmp[i+3];
            ciphertext[i]   = mul2(tmp[i]   ^ tmp[i+1]) ^ tmp[i]   ^ t;
            ciphertext[i+1] = mul2(tmp[i+1] ^ tmp[i+2]) ^ tmp[i+1] ^ t;
            ciphertext[i+2] = mul2(tmp[i+2] ^ tmp[i+3]) ^ tmp[i+2] ^ t;
            ciphertext[i+3] = mul2(tmp[i+3] ^ tmp[i]  ) ^ tmp[i+3] ^ t;
        }

        // AddRoundKey
        for ( i = 0; i < AES_BLOCK_SIZE; ++i ) {
            *(ciphertext+i) ^= *roundkeys++;
        }

    }
    
    // last round
    for (i = 0; i < AES_BLOCK_SIZE; ++i) {
        *(ciphertext+i) = S_gpu[*(ciphertext+i)];
    }
    shift_rows(ciphertext);
    for ( i = 0; i < AES_BLOCK_SIZE; ++i ) {
        *(ciphertext+i) ^= *roundkeys++;
    }

}

__global__ void aes_encrypt_gpu(uint8_t *plaintext,uint8_t *ciphertext,const uint8_t *roundkeys,unsigned int total_blocks) {
//__global__ void aes_encrypt_gpu(uint8_t *plaintext,uint8_t *ciphertext,unsigned int total_blocks) {
	
	unsigned int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int data_block_offset = thread_id*AES_BLOCK_SIZE;
/*
	uint8_t key[] = {
		//0x0f, 0x15, 0x71, 0xc9, 0x47, 0xd9, 0xe8, 0x59, 
		//0x0c, 0xb7, 0xad, 0xd6, 0xaf, 0x7f, 0x67, 0x98,
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 

	};

	uint8_t roundkeys[AES_ROUND_KEY_SIZE];
*/
	if(thread_id<total_blocks) {

//		aes_key_schedule_128_gpu(key, roundkeys);
		aes_encrypt_128_gpu(roundkeys, &plaintext[data_block_offset], &ciphertext[data_block_offset]);

	}

}



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

	uint8_t plaintext[] = {
		//0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
		//0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
		0x11, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
		0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
	};

 
	uint8_t ciphertext[AES_BLOCK_SIZE];

	const uint8_t const_cipher[AES_BLOCK_SIZE] = {
		//0xff, 0x0b, 0x84, 0x4a, 0x08, 0x53, 0xbf, 0x7c,
		//0x69, 0x34, 0xab, 0x43, 0x64, 0x14, 0x8f, 0xb9,
		0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30,
		0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a,
	};
	
	uint8_t roundkeys[AES_ROUND_KEY_SIZE];
	uint8_t *plaintext_gpu, *ciphertext_gpu, *roundkeys_gpu;
	uint8_t *plaintext_cpu, *ciphertext_cpu;

	struct timespec start_read, end_read;
	uint64_t  read_time;
	
	cudaEvent_t start,stop;
	cudaEvent_t stop_dtrans_gpu;  //To track CPU to GPU data transfer
	cudaEvent_t start_dtrans_cpu; //To track GPU to CPU data transfer
	float total_time;	
	float total_dtrans_gpu_time;
	float total_dtrans_cpu_time;
	float kernel_execution_time;

 
  	unsigned int total_blocks=0;
	unsigned int total_threads=0;
	unsigned int thread_blocks;
	unsigned int size;
	total_blocks = atoi(argv[2]);
	total_threads=total_blocks;
	
	size=AES_BLOCK_SIZE*total_blocks;

	plaintext_cpu = (uint8_t*)malloc(size*sizeof(uint8_t));
	ciphertext_cpu = (uint8_t*)malloc(size*sizeof(uint8_t));
	
	clock_gettime(CLOCK_MONOTONIC,&start_read);
	FILE *fptr = fopen(argv[1],"rb");
	fread(plaintext_cpu,size,1,fptr);

	clock_gettime(CLOCK_MONOTONIC,&end_read);
	read_time = BILLION*(end_read.tv_sec-start_read.tv_sec) + (end_read.tv_nsec - start_read.tv_nsec);
	printf("Time taken to read the file in nano seconds is  %lu \n" , read_time);

	aes_key_schedule_128(key, roundkeys);

        thread_blocks=(total_threads/THREADS_PER_BLOCK) + 1;

	cudaEventCreate(&start);
        cudaEventCreate(&stop);
       	cudaEventCreate(&start_dtrans_cpu);
	cudaEventCreate(&stop_dtrans_gpu);

	cudaEventRecord(start, 0);
        cudaMalloc((void**)&plaintext_gpu,size*sizeof(uint8_t));
        cudaMalloc((void**)&ciphertext_gpu,size*sizeof(uint8_t));
        cudaMalloc((void**)&roundkeys_gpu,AES_ROUND_KEY_SIZE*sizeof(uint8_t));


	cudaErrorCheck(cudaMemcpyToSymbol(S_gpu,SBOX,256*sizeof(uint8_t)));
	cudaErrorCheck(cudaMemcpyToSymbol(RC_gpu,RC,10*sizeof(uint8_t)));

        cudaErrorCheck(cudaMemcpy(plaintext_gpu,plaintext_cpu,size*sizeof(uint8_t),cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy(roundkeys_gpu,roundkeys,AES_ROUND_KEY_SIZE*sizeof(uint8_t),cudaMemcpyHostToDevice));

	cudaEventRecord(stop_dtrans_gpu,0);

//        printf("Thread blocks =%u, total_threads = %u, total_text_blocks = %u\n",thread_blocks,total_threads, total_blocks);
	aes_encrypt_gpu<<<thread_blocks,THREADS_PER_BLOCK>>>(plaintext_gpu,ciphertext_gpu,roundkeys_gpu,total_blocks) ;
	//aes_encrypt_gpu<<<thread_blocks,THREADS_PER_BLOCK>>>(plaintext_gpu,ciphertext_gpu,total_blocks) ;

        cudaErrorCheck(cudaPeekAtLastError());
	
	cudaEventRecord(start_dtrans_cpu,0);
        cudaErrorCheck(cudaMemcpy(ciphertext_cpu,ciphertext_gpu,size*sizeof(uint8_t),cudaMemcpyDeviceToHost));

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
	cudaFree(ciphertext_gpu);
	cudaFree(plaintext_gpu);
	cudaFree(roundkeys_gpu);
	fclose(fptr);
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
