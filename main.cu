
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 32

/*
struct Drones
{
float *x;
float *y;
float *dx;
float *dy;
float *nx;
float *ny;

} *drones;
*/


__global__ void setupFlight(float *x, float *y, float *dx, float *dy, float *nx, float *ny, int n, unsigned int seed, curandState_t* states)
{
	
	/* we have to initialize the state */
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		threadIdx.x, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[threadIdx.x]);

	int i = threadIdx.x;


	if (i < n)
	{
		//calculate initial pos and velocity
		x[i] = curand(&states[threadIdx.x]) % 129;
		y[i] = curand(&states[threadIdx.x]) % 129;

		//calculate random speed
		dx[i] = (curand(&states[threadIdx.x]) % 601 + 30);
		dy[i] = (curand(&states[threadIdx.x]) % 601 + 30);
		//knots per half second
		dx[i] = (dx[i] / 3600) * 0.5;
		dy[i] = (dy[i] / 3600) * 0.5;

		nx[i] = x[i] + dx[i];
		ny[i] = y[i] + dy[i];

		/*
		x[i] += dx[i];
		y[i] += dy[i];
		*/
		
		if (nx[i] >= 128.00)
		{
			nx[i] = nx[i] * (-1.00);
		}
		if (ny[i] >= 128.00)
		{
			ny[i] = ny[i] * (-1.00);
		}

		

	}
}


int main(void)
{
	//Drones *drone_h, *drone_d;
	//init random numbers
	//srand(time(NULL));
	//set up host copies
	float *x, *y, *dx, *dy, *nx, *ny;
	//set up device copies
	float *d_x, *d_y, *d_dx, *d_dy, *d_nx, *d_ny;
	int size = sizeof(float) * N;

	/* CUDA's random number library uses curandState_t to keep track of the seed value
	we will store a random state for every thread  */
	curandState_t* states;

	/* allocate space on the GPU for the random states */
	cudaMalloc((void**)&states, N * sizeof(curandState_t));

	//allocate host memory
	x = (float*)malloc(size);
	y = (float*)malloc(size);
	dx = (float*)malloc(size);
	dy = (float*)malloc(size);
	nx = (float*)malloc(size);
	ny = (float*)malloc(size);

	//allocate device memory
	cudaMalloc((void**)&d_x, size);
	cudaMalloc((void**)&d_y, size);
	cudaMalloc((void**)&d_dx, size);
	cudaMalloc((void**)&d_dy, size);
	cudaMalloc((void**)&d_nx, size);
	cudaMalloc((void**)&d_ny, size);

	//cudaMalloc((void **)&drone_d, sizeof(Drones)*N);
	//drone_h = (Drones *)malloc(sizeof(Drones)*N);

	
	
	//initiate arrays
	int i;
	for(i = 0; i < N; i++)
	{
		x[i] = 0;
		y[i] = 0;
		dx[i] = 0;
		dy[i] = 0;
		nx[i] = 0;
		ny[i] = 0;
	}

	//*dx = ((rand() % 601 + 30) / 3600) * 0.5;
	//*dy = ((rand() % 601 + 30) / 3600) * 0.5;
	
	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, dx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dy, dy, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nx, nx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ny, ny, size, cudaMemcpyHostToDevice);

	int blocks = 1;
	setupFlight<<<blocks, N >>>(d_x, d_y, d_dx, d_dy, d_nx, d_ny, N, time(NULL), states);


	cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dx, d_dx, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dy, d_dy, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(nx, d_nx, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(ny, d_ny, size, cudaMemcpyDeviceToHost);

	printf("Initializing drones...");

	for(int j = 0; j < N; j++)
	{
		printf("Drone #:%d x: %f, y: %f, dx: %f, dy: %f\n", j, x[j], y[j], dx[j], dy[j]);
	}
	
	//radar and collision detection
	//new kernel function to take all these values and keep adding their dx/dy and 
	//checking nx/ny with 2 nautical mile bounding box (+2/-2 on x and y from current x and y)

	free(x); free(y); free(dx); free(dy); free(nx); free(ny);

	cudaFree(d_x); cudaFree(d_y); cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_nx); cudaFree(d_ny);

	return 0;
}
