__kernel  void gpu_psearch(__global unsigned int *list, __global unsigned int *query,__global unsigned char *present, const unsigned int total_queries, const unsigned int list_size) {

	//unsigned int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int thread_id = get_global_id(0);
	unsigned int i;
	if(thread_id < total_queries) {
		present[thread_id]=0;
		for (i=0;i<list_size;i++) {
			if(query[thread_id]==list[i])
				present[thread_id]=1;
		}
			

	}	
	
}
