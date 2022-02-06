#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  //   and launch your kernel from the main function
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if (tid >= len) 
    {
        return;
    }
    printf("my thread id is %d and block id is %d\n",threadIdx.x, blockIdx.x);
    out[tid] = in1[tid] + in2[tid];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  if (cudaMalloc((void **) &deviceInput1, inputLength*sizeof(float)) != cudaSuccess) {
      printf("malloc error for deviceInput1\n");
      return 0;
  }
  if (cudaMalloc((void **) &deviceInput2, inputLength*sizeof(float)) != cudaSuccess) {
      printf("malloc error for deviceInput2\n");
      return 0;
  }
  if (cudaMalloc((void **) &deviceOutput, inputLength*sizeof(float)) != cudaSuccess) {
      printf("malloc error for deviceOutput\n");
      return 0;
  }

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  if (cudaMemcpy(deviceInput1,hostInput1,inputLength*sizeof(float),cudaMemcpyHostToDevice) != cudaSuccess){
      cudaFree(deviceInput1);
      cudaFree(deviceInput2);
      cudaFree(deviceOutput);
      free(hostInput1);
      free(hostInput2);
      free(hostOutput);
      printf("data transfer error from host to device on deviceInput1\n");
      return 0;
  }
  if (cudaMemcpy(deviceInput2,hostInput2,inputLength*sizeof(float),cudaMemcpyHostToDevice) != cudaSuccess){
      cudaFree(deviceInput1);
      cudaFree(deviceInput2);
      cudaFree(deviceOutput);
      free(hostInput1);
      free(hostInput2);
      free(hostOutput);
      printf("data transfer error from host to device on deviceInput2\n");
      return 0;
  }

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 mygrid(ceil(inputLength/256.0));
  dim3 myblock(256);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  vecAdd<<<mygrid,myblock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  if (cudaMemcpy(hostOutput,deviceOutput,inputLength*sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess){
      printf("data transfer error from device to host on deviceOutput\n");
  }

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
