#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<mpi.h>
#include<assert.h>
#include<string.h>


extern bool HL_kernelLaunch(unsigned char* d_data, unsigned char* d_resultData, size_t worldWidth, size_t worldHeight, ushort threadsCount, int myrank);

static inline unsigned char* HL_initAllZeros( size_t worldWidth, size_t worldHeight )
{

    size_t dataLength = worldWidth * worldHeight;
    unsigned char *data = NULL;
    // calloc init's to all zeros
    data = calloc( dataLength, sizeof(unsigned char));

    return data;
}

static inline unsigned char* HL_initAllOnes( size_t worldWidth, size_t worldHeight )
{

    size_t dataLength = worldWidth * worldHeight;
    // Current state of world. 
    unsigned char *data=NULL;
    size_t i;
    
    data = calloc( dataLength, sizeof(unsigned char));

    // set all rows of world to true
    for( i = 0; i < dataLength; i++)
    {
	data[i] = 1;
    }
    
    return data;
}

static inline unsigned char* HL_initOnesInMiddle( size_t worldWidth, size_t worldHeight )
{


    size_t dataLength = worldWidth * worldHeight;
    // Current state of world. 
    unsigned char *data=NULL;
    int i;
    
    data = calloc( dataLength, sizeof(unsigned char));

    // set first 1 rows of world to true
    for( i = 10 * worldWidth; i < 11 * worldWidth; i++)
    {
	if( (i >= ( 10 * worldWidth + 10)) && (i < ( 10 * worldWidth + 20)))
	{
	    data[i] = 1;
	}
    }
    return data;    
}

static inline unsigned char* HL_initOnesAtCorners( size_t worldWidth, size_t worldHeight )
{
    size_t dataLength = worldWidth * worldHeight;
    // Current state of world.
    unsigned char *data=NULL;

    data = calloc( dataLength, sizeof(unsigned char));

    data[0] = 1; // upper left
    data[worldWidth-1]=1; // upper right
    data[(worldHeight * (worldWidth-1))]=1; // lower left
    data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
    return data;
}

static inline unsigned char* HL_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight )
{
    size_t dataLength = worldWidth * worldHeight;
    // Current state of world.
    unsigned char *data=NULL;

    data = calloc( dataLength, sizeof(unsigned char));

    data[0] = 1; // upper left
    data[1] = 1; // upper left +1
    data[worldWidth-1]=1; // upper right
    return data;
}

static inline unsigned char* HL_initReplicator( size_t worldWidth, size_t worldHeight )
{
    size_t dataLength = worldWidth * worldHeight;
    // Current state of world.
    unsigned char *data=NULL;

    size_t x, y;
    
    data = calloc( dataLength, sizeof(unsigned char));

    x = worldWidth/2;
    y = worldHeight/2;
    
    data[x + y*worldWidth + 1] = 1; 
    data[x + y*worldWidth + 2] = 1;
    data[x + y*worldWidth + 3] = 1;
    data[x + (y+1)*worldWidth] = 1;
    data[x + (y+2)*worldWidth] = 1;
    data[x + (y+3)*worldWidth] = 1; 

    return data;
    
}

static inline unsigned char* HL_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight )
{
    
    switch(pattern)
    {
    case 0:
	return HL_initAllZeros( worldWidth, worldHeight );
	break;
	
    case 1:
	return HL_initAllOnes( worldWidth, worldHeight );
	break;
	
    case 2:
	return HL_initOnesInMiddle( worldWidth, worldHeight );
	break;
	
    case 3:
	return HL_initOnesAtCorners( worldWidth, worldHeight );
	break;

    case 4:
	return HL_initSpinnerAtCorner( worldWidth, worldHeight );
	break;

    case 5:
	return HL_initReplicator( worldWidth, worldHeight );
	break;
	
    default:
	printf("Pattern %u has not been implemented \n", pattern);
	exit(-1);
    }
    

}

static inline void HL_swap( unsigned char **pA, unsigned char **pB)
{
  unsigned char *temp = *pA;
  *pA = *pB;
  *pB = temp;
}
 

// Don't Modify this function or your submitty autograding will not work
static inline void HL_printWorld(size_t iteration, size_t worldHeight, size_t worldWidth, unsigned char *data)
{
    int i, j;

    printf("Print World - Iteration %lu \n", iteration);
    
    for( i = 0; i < worldHeight; i++)
    {
	printf("Row %2d: ", i);
	for( j = 0; j < worldWidth; j++)
	{
	    printf("%u ", (unsigned int)data[(i*worldWidth) + j]);
	}
	printf("\n");
    }

    printf("\n\n");
}


int main(int argc, char *argv[])
{
    
   
    int pattern, worldSize , iterations, world_rank, numranks, worldRow_per_proc, dataLength_per_proc, threadBlockSize,cE, cudaDeviceCount;
    double tstart, tend;    


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    
    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    threadBlockSize = atoi(argv[4]);
    worldRow_per_proc = worldSize/ numranks;
    dataLength_per_proc = worldRow_per_proc * worldSize;
   
    bool output = argc > 5 && strcmp(argv[5],"true") == 0; 

    unsigned char* slice_data =  calloc((worldRow_per_proc + 2) * worldSize, sizeof(unsigned char));
    unsigned char* slice_resultData = calloc((worldRow_per_proc + 2) * worldSize, sizeof(unsigned char));
    assert(slice_data != NULL); 
    assert(slice_resultData != NULL);
    // Initialise and scatter the data in all the processes      
    
    unsigned char* g_data = NULL;
    if (world_rank == 0){
        tstart = MPI_Wtime();
        g_data = HL_initMaster(pattern, worldSize, worldSize);
    //	printf("AFTER INITILIZATION WORLD IS............\n");
    //    HL_printWorld(0, worldSize, worldSize, g_data);	
        MPI_Scatter(g_data, dataLength_per_proc, MPI_UNSIGNED_CHAR, slice_data + worldSize,
		   dataLength_per_proc, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);   
    } else {
        
        MPI_Scatter(NULL, 0, MPI_UNSIGNED_CHAR, slice_data + worldSize,
	   dataLength_per_proc, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    }	    

 
     // Send to the previous rank and receive from the next rank
    int prev_rank = (world_rank - 1 + numranks) % numranks;
    int next_rank = (world_rank + 1) % numranks;

    for (int i = 0; i < iterations; ++i){
          // Buffer for receiving data from the previous and next rank
        //   unsigned char *recv_from_prev = slice_data;  // Top ghost row
        //   unsigned char *recv_from_next = slice_data + (worldRow_per_proc + 1) * worldSize;  // Bottom ghost row

            // Buffer for sending data to the previous and next rank
        //   unsigned char *send_to_prev = slice_data + worldSize;  // First row (row after ghost row)
       //   unsigned char *send_to_next = slice_data + worldRow_per_proc * worldSize;  // Last row (not ghost row, just second last row)

       MPI_Request requests[4];
       MPI_Status statuses[4];



       MPI_Isend(slice_data + worldSize, worldSize, MPI_UNSIGNED_CHAR, prev_rank, 0, MPI_COMM_WORLD, &requests[0]);
       MPI_Irecv(slice_data + (worldRow_per_proc+1) * worldSize, worldSize, MPI_UNSIGNED_CHAR, next_rank, 0, MPI_COMM_WORLD, &requests[1]);

       // Send to the next rank and receive from the previous rank
       MPI_Isend(slice_data + worldRow_per_proc * worldSize, worldSize, MPI_UNSIGNED_CHAR, next_rank, 0, MPI_COMM_WORLD, &requests[2]);
       MPI_Irecv(slice_data, worldSize, MPI_UNSIGNED_CHAR, prev_rank, 0, MPI_COMM_WORLD, &requests[3]);

       // Wait for all non-blocking operations to complete
       int err = MPI_Waitall(4, requests, statuses);
       if (err != MPI_SUCCESS) {
           char error_string[BUFSIZ];
           int length_of_error_string, error_class;

           MPI_Error_class(err, &error_class);
           MPI_Error_string(error_class, error_string, &length_of_error_string);
           printf("%3d: %s\n", world_rank, error_string);

           MPI_Error_string(err, error_string, &length_of_error_string);
           printf("%3d: %s\n", world_rank, error_string);

           MPI_Abort(MPI_COMM_WORLD, err);
       }
      

        
       HL_kernelLaunch(slice_data, slice_resultData, worldSize, worldRow_per_proc + 2, threadBlockSize, world_rank);
       HL_swap(&slice_data, &slice_resultData);
   }   
       MPI_Barrier(MPI_COMM_WORLD);
   
   if (world_rank == 0){
   	tend = MPI_Wtime();
 	printf("Execution time = %f", tend-tstart);
   }
   
    if (output) {
        if (world_rank == 0) {
          MPI_Gather(slice_data + worldSize, dataLength_per_proc, MPI_UNSIGNED_CHAR, g_data, dataLength_per_proc, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        } else {
          MPI_Gather(slice_data + worldSize, dataLength_per_proc, MPI_UNSIGNED_CHAR, NULL, 0, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        }
    }	
   
    
    if (world_rank == 0) 
    {
    printf("######################### FINAL WORLD IS ###############################\n");
    HL_printWorld(iterations, worldSize, worldSize, g_data);
    free(g_data);
    }
    
    free(slice_data);
    free(slice_resultData);
    
    MPI_Finalize();
    
    return 0;
}
