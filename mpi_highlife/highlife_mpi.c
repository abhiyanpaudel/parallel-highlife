/*** Abhiyan Paudel *****/
/*** Assignment 3 ******/
/***********************/

// This is the HighLife implementation in MPI. The code is based on the serial implementation of HighLife.

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<mpi.h>
#include<assert.h>
#include<string.h>


// Function to initialize the world with all cells dead 
static inline unsigned char* HL_initAllZeros( size_t worldWidth, size_t worldHeight )
{

    size_t dataLength = worldWidth * worldHeight;
    unsigned char *data = NULL;
    // calloc init's to all zeros
    data = calloc( dataLength, sizeof(unsigned char));

    return data;
}

// Function to initialize the world with all cells alive
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


// Function to initialize the world with a row of  alive cells in the middle
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


// Function to initialize the world with alive cells at the corners
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

// Function to initialize the world with a spinner at the corner

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


// Function to initialize the world with a replicator

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


// Function to initialize the world based on the pattern

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

// Function to swap the pointers

static inline void HL_swap( unsigned char **pA, unsigned char **pB)
{
  unsigned char *temp = *pA;
  *pA = *pB;
  *pB = temp;
}

// Function to count the number of alive cells

static inline unsigned int HL_countAliveCells(unsigned char* data, 
					   size_t x0, 
					   size_t x1, 
					   size_t x2, 
					   size_t y0, 
					   size_t y1,
					   size_t y2) 
{
  
  return data[x0 + y0] + data[x1 + y0] + data[x2 + y0]
    + data[x0 + y1] + data[x2 + y1]
    + data[x0 + y2] + data[x1 + y2] + data[x2 + y2];
}

// Function to print the world

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

// Function to update the world based on the rules of HighLife 

bool HL_worldUpdate(size_t worldHeight, size_t worldWidth, unsigned char *data, unsigned char *resultData)
{
 
     size_t i, y, x;

     for (y = 1; y < worldHeight - 1; ++y) 
	{
	  size_t y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
	  size_t y1 = y * worldWidth;
	  size_t y2 = ((y + 1) % worldHeight) * worldWidth;
	  
     for (x = 0; x < worldWidth; ++x) 
	  {
	    size_t x0 = (x + worldWidth - 1) % worldWidth;
	    size_t x2 = (x + 1) % worldWidth;
	  
	    unsigned int aliveCells = HL_countAliveCells(data, x0, x, x2, y0, y1, y2);
	    resultData[x + y1] = (aliveCells == 3) || (aliveCells == 6 && !data[x + y1])
	      || (aliveCells == 2 && data[x + y1]) ? 1 : 0;
	}
      }
    
  return true;
}

int main(int argc, char *argv[])
{
    
   
    int pattern, worldSize , iterations, world_rank, numranks, worldRow_per_proc, dataLength_per_proc; 
    double tstart, tend;    


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    //printf("This is the HighLife running in parallel on a CPU.\n");

    if( argc != 5 ) {
        if (world_rank == 0) {
		printf("HighLife requires 4 arguments, 1st is pattern number, 2nd the sq size of the world and 3rd is the number of itterations,
		4th is true or false for gathering in root process  e.g. ./highlife 0 32 2 true \n");
		exit(-1);
        }
        MPI_Finalize();
        exit(-1);		
    }

    pattern = atoi(argv[1]);     
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    worldRow_per_proc = worldSize/ numranks;
    dataLength_per_proc = worldRow_per_proc * worldSize;
    
     
    bool output = argc > 4 && strcmp(argv[4],"true") == 0;
    
   // allocate memory for the slice data and slice result data 
    unsigned char* slice_data =  calloc((worldRow_per_proc + 2) * worldSize, sizeof(unsigned char));
    unsigned char* slice_resultData = calloc((worldRow_per_proc + 2) * worldSize, sizeof(unsigned char));
    assert(slice_data != NULL); 
    assert(slice_resultData != NULL);

    // Initialise and scatter the data in all the processes      
    
    unsigned char* g_data = NULL;
    if (world_rank == 0){
        tstart = MPI_Wtime();
        g_data = HL_initMaster(pattern, worldSize, worldSize); // initialising the world in root process based on the pattern
    //	printf("AFTER INITILIZATION WORLD IS............\n");
    //  HL_printWorld(0, worldSize, worldSize, g_data);	
        MPI_Scatter(g_data, dataLength_per_proc, MPI_UNSIGNED_CHAR, slice_data + worldSize,
		   dataLength_per_proc, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);   
    } else {
        
        MPI_Scatter(NULL, 0, MPI_UNSIGNED_CHAR, slice_data + worldSize,
	   dataLength_per_proc, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    }	    

 
    int prev_rank = (world_rank - 1 + numranks) % numranks;
    int next_rank = (world_rank + 1) % numranks;

    for (int i = 0; i < iterations; ++i){

       MPI_Request requests[4];
       MPI_Status statuses[4];
 

     // Send to the previous rank and receive from the next rank
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
      

    // Update the world in each process

       HL_worldUpdate(worldRow_per_proc+2, worldSize, slice_data, slice_resultData); 

   // Swap the data and resultData pointers
       HL_swap(&slice_data, &slice_resultData);

   }   
       MPI_Barrier(MPI_COMM_WORLD); // for synchronization
   
   if (world_rank == 0){
   	tend = MPI_Wtime();
// 	printf("Execution time = %f\n", tend-tstart);
   }
   
   // Gather the data from all the processes to the root process
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
        HL_printWorld(iterations, worldSize, worldSize, g_data); // print the final world in the root process
        free(g_data);
    }
    
    free(slice_data);
    free(slice_resultData);
    
    MPI_Finalize();
    
    return 0;
}
