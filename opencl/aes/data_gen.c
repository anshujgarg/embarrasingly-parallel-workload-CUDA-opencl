#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>

int main(int argc, char *argv[]) {
    printf("Usage ./data_gen <filename>  <total_blocks>\n");
    unsigned int block_size=0, total_blocks=0;
    int filename_length = strlen(argv[1]) + strlen(".bin");
    unsigned int total_data_size;
    char *filename = malloc(filename_length+1);
    strcpy(filename,argv[1]);
    strcat(filename,".bin");
    total_blocks=atoi(argv[2]);
    //block_size = atoi(argv[3]);
    block_size=16;    //128 bytes in case 
    total_data_size = total_blocks*block_size;
    printf("File name is %s\n", filename);
    uint8_t *data;
    uint8_t *read_data;
    data= (uint8_t*)malloc(total_data_size*sizeof(uint8_t));
    read_data= (uint8_t*)malloc(total_data_size*sizeof(uint8_t));
    FILE *fptr = fopen(filename,"wb");
    int i=0,j=0;
//    uint8_t a=0x12;
    uint8_t plaintext[] = {
		//0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
		//0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
		0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
		0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
	};


    for (i=0;i<total_blocks;i++) {
	    for(j=0;j<block_size;j++) {
		    memcpy(&data[i*block_size+j],&plaintext[j],1);
	    }
//    	    a=a+1;
//	    if(a>0x99)
//	    	a=0x12;
	

    }
    fwrite(data,total_data_size,1, fptr);
    fclose(fptr);

    
    FILE *fptr_read = fopen(filename,"rb");
    fread(read_data,total_data_size,1,fptr_read);
    fclose(fptr_read);

   /*
    for (i=0;i<total_blocks;i++) {
	 //   a='A';
	    for(j=0;j<block_size;j++) {
	//	    memcpy(&data[i*block_size+j],&a,1);
		    printf("%x \n", read_data[i*block_size + j]);
	    }

    }
    printf("\n");
 
*/	    
    return 0;
 }
