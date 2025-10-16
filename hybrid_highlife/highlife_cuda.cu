/*** Abhiyan Paudel ****
**********************/
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<cuda.h>
#include<cuda_runtime.h>

// CUDA kernel to calculate the next state of the world

extern "C"
{
 bool HL_kernelLaunch(unsigned char** d_data, unsigned char** d_resultData, size_t worldWidth, size_t worldHeight, ushort threadsCount, int myrank);   // function prototype
}


__global__ void HL_kernel(const unsigned char* d_data, size_t worldWidth, size_t worldHeight, int myrank, unsigned char* d_resultData);  // kernel function prototype


//  Launch the CUDA kernel to calculate the next state of the world until last iterations 
bool HL_kernelLaunch(unsigned char* g_data, unsigned char* g_resultData, size_t worldWidth, size_t worldHeight, ushort threadsCount, int myrank)
{
    
    cudaSetDevice(myrank % 4);  // set the device to be used by the current rank
    
    unsigned char *d_data, *d_resultData;	
    cudaMallocManaged(&g_data, worldHeight * worldWidth * sizeof(unsigned char));   
    cudaMallocManaged(&g_resultData, worldHeight * worldWidth * sizeof(unsigned char));
        
    cudaMemcpy(d_data, g_data, worldHeight * worldWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_resultData, g_resultData, worldHeight * worldWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);    

    
    size_t blockGridSize = (worldWidth * worldHeight + threadsCount - 1) / threadsCount;  // no. of blocks in a grid  
    
    HL_kernel<<<blockGridSize, threadsCount>>>(d_data, worldWidth, worldHeight, myrank, d_resultData);  // invoke kernel function 
    cudaMemcpy(g_data, d_data, worldHeight * worldWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

   
    cudaMemcpy(g_resultData, d_resultData, worldHeight * worldWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_resultData);
    return true;
}


__global__ void HL_kernel(const unsigned char* d_data, size_t worldWidth, 
                          size_t worldHeight, unsigned char* d_resultData){

   int device; 
   cudaGetDevice(&device);
   
   size_t index; 
   index  = threadIdx.x + blockIdx.x * blockDim.x;
 //  stride = blockDim.x * gridDim.x;

   if (index < worldWidth * worldHeight){
	 size_t x = index % worldWidth;
         size_t y = index / worldWidth;	

 	 size_t x0 = (x + worldWidth -1) % worldWidth;
	 size_t x1 = x;
	 size_t x2 = (x + 1) % worldWidth;
         
	 size_t y0, y1, y2;
         unsigned int aliveCells;		 
	 if ( y >=1 && y < worldHeight -1){
         y0 = ((y + worldHeight -1) % worldHeight) * worldWidth;    
         y1 = y * worldWidth;
         y2 = ((y + 1) % worldHeight) * worldWidth;
	 
	 aliveCells = d_data[x0 + y0] + d_data[x1 + y0] + d_data[x2 + y0]                    // calculate total number of alive neighboring cells
		                     + d_data[x0 + y1] + d_data[x2 + y1] + d_data[x0 + y2] + d_data[x1 + y2] + d_data[x2 + y2];
        
	 d_resultData[x1 + y1] = (aliveCells == 3) || (aliveCells == 6 && !d_data[x1 + y1]) // update the state of each cell 
	 || (aliveCells == 2 && d_data[x1 + y1]) ? 1 : 0;	 
         }                                                    
   }                        
}                          

