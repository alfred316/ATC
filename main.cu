#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512


__global__ void flight(float *x, float *y, float *dx, float *dy, int n)
{
	int i = threadIdx.x;
	
	
	if(i < n)
	{
		x[i] += dx[i];
		y[i] += dy[i];
		if(x[i] >= 128.00)
		{
			x[i] = x[i] * (-1.00);
		}		
		if(y[i] >= 128.00)
		{
			y[i] = y[i] * (-1.00);
		}
		
	}
}


int main (void)
{
	//init random numbers
	srand(time(NULL));
	//set up host copies
	float *x, *y, *dx, *dy;	
	//set up device copies
	float *d_x, *d_y, *d_dx, *d_dy;
	int size = sizeof(float);

	//allocate device memory
	cudaMalloc((void **)&d_x, size);
	cudaMalloc((void **)&d_y, size);
	cudaMalloc((void **)&d_dx, size);
	cudaMalloc((void **)&d_dy, size);
	
	//allocate host memory
	x = (float *)malloc(size);
	y = (float *)malloc(size);
 	dx = (float *)malloc(size);
	dy = (float *)malloc(size);
	*x = rand() % 129; 
	*y = rand() % 129;
	*dx = ((rand() % 601 + 30)/3600) * 0.5;
	*dy = ((rand() % 601 + 30)/3600) * 0.5;
	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, dx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dy, dy, size, cudaMemcpyHostToDevice);
	
	flight<<<1, N>>>(d_x, d_y, d_dx, d_dy, N);

	free(x); free(y); free(dx); free(dy);

	cudaFree(d_x); cudaFree(d_y); cudaFree(d_dx); cudaFree(d_dy);

	return 0;
}
