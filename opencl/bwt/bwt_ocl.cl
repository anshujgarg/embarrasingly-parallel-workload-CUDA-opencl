#define STRING_LENGTH 10
// Structure to store data of a rotation 
struct rotation { 
	int index; 
	__global char* suffix; 
}; 

int strcmp_gpu (__global const char * s1, __global const char * s2) {
	for(; *s1 == *s2; ++s1, ++s2)
       	if(*s1 == 0)
	        return 0;
	return *(__global unsigned char *)s1 < *(__global unsigned char *)s2 ? -1 : 1;
	
	
}




// Compares the rotations and 
// sorts the rotations alphabetically 
int cmpfunc_gpu(void* x, void* y) 
{ 
	struct rotation* rx = (struct rotation*)x; 
	struct rotation* ry = (struct rotation*)y; 
	return strcmp_gpu(rx->suffix, ry->suffix); 
}


void memcpy(void *dest, void *source, size_t size) {
	int i;
	char *src = (char *)source;
	char *dst = (char *)dest;
	for(i=0;i<size;i++)
		dst[i]=src[i];
		
}


/* you can also write compare function for floats, 
	chars, double similarly as integer. */
// function for swap two elements 
void swap_gpu(void* v1, void* v2, int size) 
{ 
	// buffer is array of characters which will 
	// store element byte by byte 
	//char buffer[size]; 
	//char *buffer = (char*)malloc(size*sizeof(char));
	char buffer[STRING_LENGTH+7];

	// memcpy will copy the contents from starting 
	// address of v1 to length of size in buffer 
	// byte by byte. 
//	if (size>STRING_LENGTH)
//		printf("size - length = %d\n",size-STRING_LENGTH);
	memcpy(buffer, v1, size); 
	memcpy(v1, v2, size); 
	memcpy(v2, buffer, size); 
	//free(buffer);
} 

/*
int partition(void* v, int size, int left, int right)
{
	void *vt, *v3; 
	int i, last, mid = (left + right) / 2; 
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
		if (cmpfunc_gpu(vl, vt) > 0) { 
			++last; //return last or last+1 in partition.
			v3 = (char*)(v + (last * size)); 
			swap_gpu(vt, v3, size); 
		} 
	} 
	v3 = (char*)(v + (last * size)); 
	swap_gpu(vl, v3, size); 
	return last;
}*/


int partition(void* v, int size, int left, int right)
{
  void *vj, *vi;
  void* vr = (char*)(v+(right*size));
  int i = (left - 1);
  for(int j=left; j <= right-1; j++)
  {
		vj = (char*)(v + (j * size)); 
    if(cmpfunc_gpu(vr,vj)>0)
    {
      i++;
			vi = (char*)(v + (i * size)); 
      swap_gpu(vi,vj,size);
    }
  }
	vi = (char*)(v + ((i+1) * size)); 
  swap_gpu(vi,vr, size);
  return (i+1);
}



void _qsort_gpu(void* v, int size, int left, int right) 
{
	//int* temp = (int*)malloc((right-left+1) * sizeof(int)); 
	int temp[STRING_LENGTH];
	//printf("%d\n",right-left+1);
	int top = -1;
	temp[++top] = left;
	temp[++top] = right;
	while(top >= 0)
	{
		right = temp[top--];
		left = temp[top--];
		int p = partition(v, size, left, right);
		if(p-1>left)
		{
			temp[++top]=left;
			temp[++top]=p-1;
		
		}
		if(p+1<right)
		{
			temp[++top]=p+1;
			temp[++top]=right;
		}
	
	}
	//free(temp);
	
}



// Takes suffix array and its size 
// as arguments and returns the 
// Burrows - Wheeler Transform of given text 
void findLastChar_gpu(__global char* input_text,__private char *bwt_arr, 
				__private int* suffix_arr, int n) 
{ 
	// Iterates over the suffix array to find 
	// the last char of each cyclic rotation 
	//char* bwt_arr = (char*)malloc(n * sizeof(char)); 
	//char bwt_arr[STRING_LENGTH];
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
	//return bwt_arr; 
} 



void computeSuffixArray_gpu(__global char* input_text, __private int *suffix_arr ,int len_text) 
{ 
	// Array of structures to store rotations and 
	// their indexes 
	//struct rotation suff[len_text];
	
	
	int i=0;

	struct rotation suff[STRING_LENGTH];
	//struct rotation *suff =(struct rotation*)malloc(len_text*sizeof(struct rotation));
	//if(suff==NULL)
	//	printf("Got null value\n");


	// Structure is needed to maintain old indexes of 
	// rotations after sorting them 

	for (i = 0; i < len_text; i++) { 
		suff[i].index = i; 
		suff[i].suffix = (input_text + i); 
	}
//	if(get_global_id(0)==0) {
//	printf("Before ");
//	for (i=0;i<10;i++)
//		printf("%c",*(suff[i].suffix));
//	printf("\n After ");
//	}
	
	_qsort_gpu(suff,sizeof(struct rotation) ,0,len_text-1); 
//	if(get_global_id(0)==0) {
//	for (i=0;i<10;i++)
//		printf("%c",*(suff[i].suffix));
//	printf("\n");
//	}
//	int* suffix_arr = (int*)malloc(len_text * sizeof(int)); 
	for (i = 0; i < len_text; i++) 
		suffix_arr[i] = suff[i].index; 

	// Returns the computed suffix array 
	//return suffix_arr; 
} 





//GPU kernel to compute BWT




__kernel void bwt_gpu_func(__global char *strings, __global char *bwt, const unsigned int string_length, const unsigned int total_strings) {

	//unsigned int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int thread_id = get_global_id(0);
	unsigned int offset = thread_id*string_length;
	unsigned int end_offset = (thread_id+1)*string_length;
	//int *suffix_arr;
	int suffix_arr[STRING_LENGTH];
	char bwt_arr[STRING_LENGTH];
	int i,j;
	if(thread_id < total_strings) {

		computeSuffixArray_gpu(&strings[offset],suffix_arr,string_length);	
		findLastChar_gpu(&strings[offset],bwt_arr,suffix_arr,string_length);
		//free(suffix_arr);
		for(i=offset,j=0;i<end_offset;i++,j++) {
			bwt[i]=bwt_arr[j];
//			bwt[i]='A';
		}
		//free(bwt_arr);

	}


		
}



