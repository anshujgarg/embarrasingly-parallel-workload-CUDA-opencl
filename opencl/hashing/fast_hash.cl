//#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)+(uint32_t)(((const uint8_t *)(d))[0]))
#define get16bits(d) ((((unsigned int)(((__global const unsigned char *)(d))[1])) << 8)+(unsigned int)(((__global const unsigned char *)(d))[0]))


__kernel  void gpu_fast_page_hash(__global char *page_data, __global unsigned int *page_hash_gpu, const  unsigned int page_size,const unsigned int total_threads) {

	//int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int thread_id = get_global_id(0);
	//uint32_t len = page_size;
	unsigned int len = page_size;
	//uint32_t hash ,tmp;
	unsigned int hash, tmp;
	__global char *temp_ptr;
	hash=len;
	int rem;

	unsigned int thread_page_index=thread_id*page_size;
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
			temp_ptr  += 2*sizeof (unsigned short);
			hash  += hash >> 11;
		}

		/* Handle end cases */

		switch (rem) {
		case 3: hash += get16bits (temp_ptr);
		hash ^= hash << 16;
		hash ^= ((signed char)temp_ptr[sizeof (unsigned short)]) << 18;
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



