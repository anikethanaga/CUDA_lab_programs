#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define N 256


__global__ void vecAdd(int *a,int *b,int *c);

void vecAdd_cpu(int a[N],int b[N],int c[N]){
  for(int i=0;i<N;i++){
    c[i]=a[i]+b[i];
  }
}

int main(){
  int a[N],b[N],gpu_add[N],cpu_add[N];
  int *dev_a, *dev_b, *dev_c;
  float time_gpu,time_cpu,timeindex,timeinit;

  for(int i=0;i<N;i++){
    a[i]=i+i;
    b[i]=i*i;
  }

   int size=N*sizeof(int);

   cudaMalloc((void**) &dev_a, size);
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

   vecAdd<<<1,N>>> (dev_a,dev_b,dev_c);

   cudaDeviceSynchronize();
 	 cudaEventRecord(gpu_end, 0);
 	 cudaEventSynchronize(gpu_end);
 	 cudaEventElapsedTime(&time_gpu, gpu_start, gpu_end);


   cudaEvent_t startindex,endindex;
   cudaEventCreate(&startindex);
 	 cudaEventCreate(&endindex);
 	 cudaEventRecord(startindex, 0);

   cudaMemcpy(gpu_add,dev_c,size,cudaMemcpyDeviceToHost);

   cudaEventRecord(endindex, 0);
 	 cudaEventSynchronize(endindex);
 	 cudaEventElapsedTime(&timeindex, startindex, endindex);

   clock_t cpu_start,cpu_end;
   cpu_start=clock();
   vecAdd_cpu(a,b,cpu_add);
   cpu_end=clock();

   timeinit/=1000;
   timeindex/=1000;
   time_gpu/=1000;
   time_cpu=float(cpu_end-cpu_start)/float(CLOCKS_PER_SEC);

   printf("Time for sending initial data from host to device : %f\t sec\n",timeinit);
   printf("Cuda program launched with 1 block and %d threads\n",N);
   printf("Time for sending calculated data from device to host : %f\t sec\n",timeindex);
   printf("GPU Time:%f seconds\n",time_gpu);
   printf("CPU Time:%f seconds\n",time_cpu);

   int flag=1;
   for(int i=0;i<N;i++){
     if(gpu_add[i]!=cpu_add[i]){
       flag=0;
       break;
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

   exit(0);

}

__global__ void vecAdd(int *a,int *b,int *c){
  int i=threadIdx.x;
  c[i]=a[i]+b[i];
}

