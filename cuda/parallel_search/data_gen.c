#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main(int argc, char *argv[]) {
    printf("Usage ./data_gen <filename>  <list_size>  <total_queries>\n");
    unsigned int list_size=0, total_blocks=0;
    int filename_length = strlen(argv[1]) + strlen(".bin");
    int query_filename_length = strlen(argv[3]) + strlen("_query.bin");
    unsigned int total_data_size;
    unsigned int total_queries;
    char *filename = malloc(filename_length+1);
    char *query_filename = malloc(query_filename_length+1);
    
    strcpy(filename,argv[1]);
    strcat(filename,".bin");
    strcpy(query_filename,argv[3]);
    strcat(query_filename,"_query.bin");
    list_size=atoi(argv[2]);
    total_queries = atoi(argv[3]);
 //   total_data_size = total_blocks*block_size;
    printf("File name is %s\n", filename);
    unsigned int *data,*query_data;
    unsigned int *read_data, *read_queries;
    data= (unsigned int*)malloc(list_size*sizeof(unsigned int));
    query_data= (unsigned int*)malloc(list_size*sizeof(unsigned int));
    read_data= (unsigned int*)malloc(total_queries*sizeof(unsigned int));
    read_queries= (unsigned int*)malloc(total_queries*sizeof(unsigned int));

    FILE *fptr = fopen(filename,"wb");
    unsigned int i=0,j=0;
    for (i=0;i<list_size;i++)
    	data[i]=i;
    fwrite(data,sizeof(unsigned int),list_size, fptr);

    FILE *query_fptr = fopen(query_filename,"wb");
    for (i=0;i<total_queries;i++)
    	query_data[i]=i;



    fwrite(query_data,sizeof(unsigned int),total_queries, query_fptr);
    fclose(fptr);
    fclose(query_fptr);
/*
    
    FILE *fptr_read = fopen(filename,"rb");
    fread(read_data,sizeof(unsigned int),list_size,fptr_read);
    fclose(fptr_read);
    for (i=0;i<list_size;i++)
    	printf("%u --  %u \t",i,read_data[i]);

    FILE *fptr_query_read = fopen(query_filename,"rb");
    fread(read_queries,sizeof(unsigned int),total_queries,fptr_query_read);
    fclose(fptr_query_read);
    for (i=0;i<total_queries;i++)
    	printf("%u --  %u \t",i,read_queries[i]);

 */
 
    /*
    for (i=0;i<total_blocks;i++) {
	 //   a='A';
	    for(j=0;j<block_size;j++) {
	//	    memcpy(&data[i*block_size+j],&a,1);
		    printf("%c", read_data[i*block_size + j]);
	    }

    }
    printf("\n");
 */
	    
    return 0;
 }
