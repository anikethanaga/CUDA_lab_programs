#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda.h>


#define BLOCK_DIM 16


__global__ void matrixMult(int *a,int *b,int *c,int width);


int main(){
  int curr=2;
int N=BLOCK_DIM*curr;
  printf("------------------------------------------\n");
  while(N<=BLOCK_DIM*16){
  int a[N][N], b[N][N], gpu_mul[N][N],cpu_mul[N][N];
  int *dev_a, *dev_b, *dev_c;
  float time_gpu,time_cpu,timeindex,timeinit;

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      a[i][j]=i+j;
      b[i][j]=i*j;
    }
  }

  int size=N*N*sizeof(int);
  cudaMalloc((void**) &dev_a,size);
  cudaMalloc((void**) &dev_b,size);
  cudaMalloc((void**) &dev_c,size);

  cudaEvent_t startinit,endinit;
  cudaEventCreate(&startinit);
  cudaEventCreate(&endinit);
  cudaEventRecord(startinit, 0);

  cudaMemcpy(dev_a,a,size,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b,b,size,cudaMemcpyHostToDevice);

  cudaEventRecord(endinit, 0);
  cudaEventSynchronize(endinit);
  cudaEventElapsedTime(&timeinit, startinit, endinit);



  cudaEvent_t gpu_start,gpu_end;
  cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_end);
	cudaEventRecord(gpu_start, 0);

  dim3 dimBlock(BLOCK_DIM,BLOCK_DIM);
  dim3 dimGrid((int)ceil(N/dimBlock.x),(int)ceil(N/dimBlock.y));

  matrixMult<<<dimGrid,dimBlock>>>(dev_a,dev_b,dev_c,N);

  cudaDeviceSynchronize();
	cudaEventRecord(gpu_end, 0);
	cudaEventSynchronize(gpu_end);
	cudaEventElapsedTime(&time_gpu, gpu_start, gpu_end);


  cudaEvent_t startindex,endindex;
  cudaEventCreate(&startindex);
	cudaEventCreate(&endindex);
	cudaEventRecord(startindex, 0);

  cudaMemcpy(gpu_mul,dev_c,size,cudaMemcpyDeviceToHost);

  cudaEventRecord(endindex, 0);
	cudaEventSynchronize(endindex);
	cudaEventElapsedTime(&timeindex, startindex, endindex);

  clock_t cpu_start,cpu_end;
  cpu_start=clock();

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      int sum=0;
      for(int k=0;k<N;k++){
        sum+=a[i][k]*b[k][j];
      }
      cpu_mul[i][j]=sum;
    }
  }

  cpu_end=clock();

  timeinit/=1000;
  timeindex/=1000;
  time_gpu/=1000;
  time_cpu=float(cpu_end-cpu_start)/float(CLOCKS_PER_SEC);

  printf("Time for sending initial data from host to device : %f\t sec\n",timeinit);
  printf("Cuda program launched with %d blocks and %d threads\n",(int)ceil(N/dimBlock.x)*(int)ceil(N/dimBlock.y),BLOCK_DIM*BLOCK_DIM);
  printf("Time for sending calculated data from device to host : %f\t sec\n",timeindex);
  printf("GPU Time:%f seconds\n",time_gpu);
  printf("CPU Time:%f seconds\n",time_cpu);

  int flag=1;
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      if(gpu_mul[i][j]!=cpu_mul[i][j]){
        flag=0;
        break;
      }
    }
  }

  if(flag){
    printf("TEST PASSED\n");
    printf("SPEED UP:%f\n",time_cpu/time_gpu);
  }
  else{
    printf("TEST FAILED\n");
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  printf("------------------------------------------\n");
  curr++;
  N=BLOCK_DIM*curr;
  }
}

__global__ void matrixMult(int *a,int *b,int *c,int width){
  int k,sum=0;

  int col=threadIdx.x+blockDim.x*blockIdx.x;
  int row=threadIdx.y+blockDim.y*blockIdx.y;

  if(col<width && row<width){
    for(k=0;k<width;k++)
      sum+=a[row*width+k]*b[k*width+col];
    c[row*width+col]=sum;
  }
}

