
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <unistd.h>

#define N 32
#define pi 3.14159265358979323846


//init the positions and velocities of the drones and also calculate their first next position
__global__ void setupFlight(float *x, float *y, float *dx, float *dy, float *alt, int *col, unsigned int seed, curandState_t* states)
{
	
	/* we have to initialize the state */
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		threadIdx.x, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[threadIdx.x]);

	int i = threadIdx.x;


	if (i < N)
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

		alt[i] = (curand(&states[threadIdx.x]) % 6001 + 3000);

		col[i] = 0;
		
	}
}

//move the aircraft by adding its velocity to it's current location
__global__ void moveDrone(float *x, float *y, float *dx, float *dy, int *col)
{
	int i = threadIdx.x;
	//create shared variables for next x and y
	//next x
	__shared__ float snx[N];
	//next y
	__shared__ float sny[N];
	//collision
	__shared__ int scol[N];
	//temp x for angle calculation
	float spx[N];
	//tmp y for angle calculation
	float spy[N];
	//sin * dx for angle calculation on x axis
	float tmp[N];
	//cos * dy for angle calculation on y axis
	float tmp0[N];

	//upper line x
	float upx[N];
	//upper line y
	float upy[N];
	//lower like x
	float lowx[N];
	//lower line y
	float lowy[N];

	//tmp x for batcher's (20 minute path)
	__shared__ float batx[N];
	//tmp y for batcher's
	__shared__ float baty[N];

	float angle[N], theta[N], cs[N], sn[N];

	//calculate shared values
	if(i < N)
	{
		//calculate next x and y
		snx[i] = x[i] + dx[i];
		sny[i] = y[i] + dy[i];
		//init the shared tmp variables
		spx[i] = 0;
		spy[i] = 0;
		tmp[i] = 0;
		tmp[i] = 0;
		//init the shared upper and lower bounds
		upx[i] = 0;
		upy[i] = 0;
		lowx[i] = 0;
		lowy[i] = 0;

		//init tmp batcher's x and y
		batx[i] = snx[i];
		baty[i] = sny[i];

		__syncthreads();

		//set collision as 0
		//scol[i] = 0;
		//if they go out of bounding grid have it come back around
		if (snx[i] >= 128.00)
		{
			snx[i] = snx[i] * (-1.00);
		}
		if (sny[i] >= 128.00)
		{
			sny[i] = sny[i] * (-1.00);
		}

		//wait for calculations to be done on all threads
		__syncthreads();

		//iterate through calculated next x and y's for all drones to see if any collisions happen
		//bounding box is 1 nautical mile on each side (so just +-1 on x and y)

		float bound = 1;
		float angleBase = 1;


				//for loop to check path up to 20 minutes ahead
				//iterate 20 minutes: 20 x 120 half seconds
				//
				__shared__ int chk[N];
				chk[i] = 0;
				__shared__ int t[N];
				for(t[i] = 0; t[i] < 5 ; t[i]++)
				{
					//every iteration check for collision, if yes, then go back to beginning, fix angle and check again up to 3 times
					if(chk[i] < 3)
					{
						//loop all Q planes and check if any other plane is in our bounding box
						//might be time consuming but will work
						//this is instead of checking against 1 plane 20 times for each device

						//iterate through the drones and compare them to the current drone this thread is handling
						//each thread does an interation on all drones, if collision occures, mark both drones as 1
						for (int p = 0; p < N; ++p)
						{
							//make sure we are not comparing the same planes with each other
							if(i != p)
							{
								
									angle[i] = angleBase + (angleBase * chk[i]);
									theta[i] = (angle[i])*((pi) / (180.0));
									cs[i] = cos(theta[i]);
									sn[i] = sin(theta[i]);

									//increase y bounding box each half second by t * dx
									//x + (t * dx) is how far x will be in t half seconds
									//upper bound
									upx[i] = batx[i] + bound + (t[i] * 120 * dx[i]);
									upy[i] = baty[i] + bound + (t[i] * 120 * dy[i]);
									//lower bound
									lowx[i] = batx[i] - bound + (t[i] * 120 * dx[i]);
									lowy[i] = baty[i] - bound + (t[i] * 120 * dy[i]);

									__syncthreads();
									//collision detection and attempt correction based on angle per half second
									if ((batx[p] <= (upx[i])) && (batx[p] >= (lowx[i])))
									{
										//move plane at angle defined in host function

										batx[i] = cs[i] * snx[i] - sn[i] * sny[i];
										baty[i] = sn[i] * snx[i] + cs[i] * sny[i];

										t[i] = -1;
										chk[i] += 1;
										//exit loop
										break;
									}

									__syncthreads();

									//detection + correction for y bounding box
									if ((baty[p] <= (upy[i])) && (baty[p] >= (lowy[i])))
									{
										//move plane at angle defined in host function

										batx[i] = cs[i] * snx[i] - sn[i] * sny[i];
										baty[i] = sn[i] * snx[i] + cs[i] * sny[i];

										t[i] = -1;
										chk[i] += 1;
										//exit loop
										break;
									}

									__syncthreads();
							}
						}

					}

				}

				//check if we corrected less than 3 times and got a good path
				//if so, give the batcher's x and y to shared x and y (with their new rotations)
				if(chk[i] < 3)
				{
					snx[i] = batx[i];
					sny[i] = baty[i];
					scol[i] = 0;
				}
				else
				{
					snx[i] = batx[i];
					sny[i] = baty[i];
					scol[i] = 1;
				}

				__syncthreads();


		//wait for calculations to be done on all threads
		__syncthreads();

		//give the regular arrays their final values for memcpy by the host
		x[i] = snx[i];
		y[i] = sny[i];
		col[i] = scol[i];

	}

	
}


int main(void)
{
	cudaEvent_t allStart, allEnd, setupStart, setupEnd, eachStepStart, eachStepEnd;
	cudaEventCreate(&allStart);
	cudaEventCreate(&allEnd);
	cudaEventCreate(&setupStart);
	cudaEventCreate(&setupEnd);
	cudaEventCreate(&eachStepStart);
	cudaEventCreate(&eachStepEnd);

	cudaEventRecord(allStart);

	
	//set up host copies
	float *x, *y, *dx, *dy, *alt;
	//set up device copies
	float *d_x, *d_y, *d_dx, *d_dy, *d_alt;
	int size = sizeof(float) * N;
	int *col, *d_col;
	
	/* CUDA's random number library uses curandState_t to keep track of the seed value
	we will store a random state for every thread  */
	curandState_t* states;

	/* allocate space on the GPU for the random states */
	cudaMalloc((void**)&states, N * sizeof(curandState_t));

	//file
	FILE *file;
	char *fileName = "drones.csv";
	file = fopen(fileName, "w+");
	fprintf(file, "Drone Id, X, Y, dX, dY, Collision\n");

	//allocate host memory
	x = (float*)malloc(size);
	y = (float*)malloc(size);
	dx = (float*)malloc(size);
	dy = (float*)malloc(size);
	alt = (float*)malloc(size);
	col = (int*)malloc(sizeof(int) * N);

	//allocate device memory
	cudaMalloc((void**)&d_x, size);
	cudaMalloc((void**)&d_y, size);
	cudaMalloc((void**)&d_dx, size);
	cudaMalloc((void**)&d_dy, size);
	cudaMalloc((void**)&d_alt, size);
	cudaMalloc((void**)&d_col, sizeof(int)*N);


	//copy over from host to device	
	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, dx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dy, dy, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_alt, alt, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_col, col, sizeof(int)*N, cudaMemcpyHostToDevice);

	//init drones positions, velocity, and first "next position"
	int blocks = 1;

	cudaEventRecord(setupStart);

	setupFlight<<<blocks, N >>>(d_x, d_y, d_dx, d_dy, d_alt, d_col, time(NULL), states);

	cudaEventRecord(setupEnd);
	cudaEventSynchronize(setupEnd);
	float setupTime;
	cudaEventElapsedTime(&setupTime, setupStart, setupEnd);
	printf("Setup Flights Time for %d drones: %f ms\n", N, setupTime);

	cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dx, d_dx, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dy, d_dy, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(col, d_col, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(alt, d_alt, size, cudaMemcpyDeviceToHost);

	printf("Initializing drones...\n");

	for(int j = 0; j < N; j++)
	{
		printf("Drone #:%d x: %f, y: %f, dx: %f, dy: %f, altitude: %f, col:%d\n", j, x[j], y[j], dx[j], dy[j], alt[j], col[j]);
	}
	
	//radar and collision detection
	//new kernel function to take all these values and keep adding their dx/dy and 
	//checking nx/ny with 1 nautical mile bounding box (+1/-1 on x and y from current x and y)

	printf("Flying drones...\n");

	//each count represents half a second
	float count = 0;
	//duration is how many seconds we want to test this for multiplied by 2
	float duration = 16;

	//loop infinitely until the duration condition is met
	//each iteration run the moveDrone kernel function to move the drone and check collision
	//make sure each iteration takes half a second
	for(;;)
	{
		//interval of half a second, which is 500ms
		float interval = 500.f;

		//time_t start = time(NULL);
		
		//kernel function and memcpy and printing

		cudaEventRecord(eachStepStart);

		moveDrone << <blocks, N >> >(d_x, d_y, d_dx, d_dy, d_col);

		cudaEventRecord(eachStepEnd);
		cudaEventSynchronize(eachStepEnd);
		float eachStepTime;
		cudaEventElapsedTime(&eachStepTime, eachStepStart, eachStepEnd);
		printf("Each Iteration of Flights Time for %d drones: %f ms\n", N, eachStepTime);

		cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(dx, d_dx, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(dy, d_dy, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(col, d_col, size, cudaMemcpyDeviceToHost);

		for (int k = 0; k < N; k++)
		{
			printf("Drone #:%d x: %f, y: %f, dx: %f, dy: %f, col: %d\n", k, x[k], y[k], dx[k], dy[k], col[k]);
			fprintf(file, "%d, %f, %f, %f, %f, %d\n", k, x[k], y[k], dx[k], dy[k], col[k]);
		}

		//time_t end = time(NULL);




		count += 1;
		if(count >= duration)
		{
			break;
		}

		//if execution done in under half second, wait the rest of the time to avoid moving
		//more than once every half second
		//elapsed and timeleft are in ms (milliseconds)
		float elapsed = eachStepTime;
		float timeLeft = interval - elapsed;
		printf("time left: %f\n", timeLeft);
		//makes sure we dont take more than 500ms to execute
		if(timeLeft > 0)
		{
			//usleep takes microseconds: 1 ms = 1000 microseconds
			usleep(timeLeft * 1000);
		}
	}

	printf("End flight...\n");

	cudaEventRecord(allEnd);
	cudaEventSynchronize(allEnd);
	float allTime;
	cudaEventElapsedTime(&allTime, allStart, allEnd);
	printf("Total Execution Time for %d flights: %f ms\n", N, allTime);

	fclose(file);

/*
	time_t progEnd = time(NULL);
	float totalElapsed = difftime(progEnd, progStart);
	printf("total execution time: %f\n", totalElapsed);
*/
	cudaEventDestroy( allStart );
	cudaEventDestroy( allEnd );
	cudaEventDestroy( setupStart );
	cudaEventDestroy( setupEnd );
	cudaEventDestroy( eachStepStart );
	cudaEventDestroy( eachStepEnd );

	free(x); free(y); free(dx); free(dy); free(alt); free(col);

	cudaFree(d_x); cudaFree(d_y); cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_alt); cudaFree(d_col);

	return 0;
}
