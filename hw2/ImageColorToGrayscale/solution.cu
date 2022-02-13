#include <wb.h> 
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT DEVICE CODE HERE
// Convert the input image with channels=numChannels to a grayscale image
// 
// \param in          2D image to be converted
// \param out         Location to store 2D grayscale image
// \param width       width of image in pixels (x-direction)
// \param height      Heigth of image in pixels (y-direction)
// \param numChannels Number of channels input image uses for encoding
__global__
void grayscaleConv(float *in, float *out, int width, int height, int numChannels) {
  // Determine row and column of the thread
  int row = blockDim.y*blockIdx.y + threadIdx.y;
  int col = blockDim.x*blockIdx.x + threadIdx.x;
  
  // For now, only operate on 3 channels since modifiers are hard-coded
  // In the future, build constant memory to deal with modifiers
  if (numChannels == 3) { 
    // Bounds checking to ensure thread index is within the image dimensions
    if (row >= 0 && row < height && col >= 0 && col < width)
    {
      int outIdx = row*width + col;  // Output image has one channel
      int inIdx = numChannels*outIdx; // Input image has numChannels channels

      out[outIdx] = in[inIdx]*0.21 // Red
              + in[inIdx+1]*0.71   // Green
              + in[inIdx+2]*0.07;  // Blue
    }
  }
}

// Also modify the main function to launch thekernel. 
int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  // Hard-coding for now to use a 2D-block with 4 warps, then dividing the
  // grid about the picture. For ocelote, each SM should be able to use 2048.
  // With these dimensions below - this would be about 4 blocks per SM
  //
  // TODO A possible optimization would be to choose a block size that could
  //      minimize unused threads by more evenly dividing the image by blocks
  //      while also maintaining the area of each block to be 256 or 4 warps.
  //      Increasing #warps/block may also help here
  dim3 mygrid(ceil(imageWidth/16.0), ceil(imageHeight/16.0));
  dim3 myblock(16,16); // 256 threads per block - 4 warps

  grayscaleConv<<<mygrid, myblock>>>(
		  deviceInputImageData, 
		  deviceOutputImageData, 
		  imageWidth, 
		  imageHeight, 
		  imageChannels); 

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
