#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main(int argc, char *argv[]) {
    printf("Usage ./data_gen <filename>  <total_blocks>  <block_size>\n");
    unsigned int block_size=0, total_blocks=0;
    int filename_length = strlen(argv[1]) + strlen(".bin");
    unsigned int total_data_size;
    char *filename = malloc(filename_length+1);
    strcpy(filename,argv[1]);
    strcat(filename,".bin");
    total_blocks=atoi(argv[2]);
    block_size = atoi(argv[3]);
    total_data_size = total_blocks*block_size;
    printf("File name is %s\n", filename);
    char *data;
    char *read_data;
    data= (char*)malloc(total_data_size*sizeof(char));
    read_data= (char*)malloc(total_data_size*sizeof(char));
    FILE *fptr = fopen(filename,"wb");
    int i=0,j=0;
    char a='A';
    char b='$';
    for (i=0;i<total_blocks;i++) {
	    for(j=0;j<block_size;j++) {
	    	    if(j!=block_size-1)
			    memcpy(&data[i*block_size+j],&a,1);
		    else{
		    	    memcpy(&data[i*block_size+j],&b,1);
			    a=a-1;
			}
		    
    	    	    a=a+1;
		    if(a>'Z')
		    	a='A';
	
	    }


    }
    fwrite(data,total_data_size,1, fptr);
    fclose(fptr);

    /*
    FILE *fptr_read = fopen(filename,"rb");
    fread(read_data,total_data_size,1,fptr_read);
    fclose(fptr_read);
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
