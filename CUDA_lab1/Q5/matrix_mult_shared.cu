#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>

#define BLOCK_SIZE 25

__global__ void gpu_shared_matrix_mul(float *a, float *b, float *gpu_mul, int n)
{
	__shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int column=blockIdx.x*blockDim.x+threadIdx.x;
	float sum=0;
	int index;

	for(int i=0;i<gridDim.x;i++)
	{
		index=row*n+i*BLOCK_SIZE+threadIdx.x;
		if(index>=n*n)
		{
			tile_a[threadIdx.y][threadIdx.x]=0;
		}
		else
		{
			tile_a[threadIdx.y][threadIdx.x]=a[index];
		}
		
		index=(i*BLOCK_SIZE+threadIdx.y)*n+column;
		if(index>=n*n)
		{
			tile_b[threadIdx.y][threadIdx.x]=0;
		}
		else
		{
			tile_b[threadIdx.y][threadIdx.x]=b[index];
		}
		__syncthreads();

		for(int k=0;k<BLOCK_SIZE;k++)
		{
			sum+=tile_a[threadIdx.y][k]*tile_b[k][threadIdx.x];
		}
		__syncthreads(); 
	}
	if(row<n && column<n)
	{
		gpu_mul[row*n+column]=sum;
	}
}

void cpu_matrix_mul(float *h_a, float *h_b, float *h_mul, int n)
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			float sum=0;
			for(int l=0;l<n;l++)
			{
				sum+=h_a[i*n+l]*h_b[l*n+j];
			}
			h_mul[i*n+j]=sum;
		}
	}
}

int main()
{
	int n;
	printf("Enter Dimension of square matrix : ");
	scanf("%d", &n);
	float size=sizeof(float)*n*n;
	
	float *h_a, *h_b, *h_c, *h_cc;
	cudaMallocHost((void**)&h_a, size);
	cudaMallocHost((void**)&h_b, size);
	cudaMallocHost((void**)&h_c, size);
	cudaMallocHost((void**)&h_cc, size);

	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			h_a[i*n+j]=1;
			h_b[i*n+j]=2;
		}
	}

	float *d_a, *d_b, *d_c;
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	dim3 dimGrid((n+BLOCK_SIZE+1)/BLOCK_SIZE, (n+BLOCK_SIZE+1)/BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	clock_t gpu_start, gpu_end;
	gpu_start=clock();
	gpu_shared_matrix_mul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
	
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	gpu_end=clock();
	float gpu_time=float(gpu_end-gpu_start)/float(CLOCKS_PER_SEC);
	printf("GPU Time : %f seconds\n", gpu_time);

	clock_t cpu_start, cpu_end;
	cpu_start=clock();
	cpu_matrix_mul(h_a, h_b, h_cc, n);
	cpu_end=clock();
	float cpu_time=float(cpu_end-cpu_start)/float(CLOCKS_PER_SEC);
	printf("CPU Time : %f seconds\n", cpu_time);

	int flag=1;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			if(h_cc[i*n+j]!=h_c[i*n+j])
			{
				flag=0;
				break;
			}
		}
	}
	
	if(flag)
	{
		printf("Success! Speedup : %f\n", cpu_time/gpu_time);
	}
	else
	{
		printf("Incorrect\n");
	}
}

