
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <unistd.h>

#define N 96
#define pi 3.14159265358979323846

/*
struct Drones
{
float *x;
float *y;
float *dx;
float *dy;
float *nx;
float *ny;

};
*/

//__constant__ float *d_x, *d_y, *d_dx, *d_dy, *d_nx, *d_ny;

//init the positions and velocities of the drones and also calculate their first next position
__global__ void setupFlight(float *x, float *y, float *dx, float *dy, float *nx, float *ny, float *alt, unsigned int seed, curandState_t* states)
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

		alt[i] = (curand(&states[threadIdx.x]) % 6001 + 3000);
	}
}

//move the aircraft by adding its velocity to it's current location
__global__ void moveDrone(float *x, float *y, float *dx, float *dy, int *col, float *cs, float *sn)
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
	__shared__ float spx[N];
	//tmp y for angle calculation
	__shared__ float spy[N];
	//sin * dx for angle calculation on x axis
	__shared__ float tmp[N];
	//cos * dy for angle calculation on y axis
	__shared__ float tmp0[N];

	//upper line x
	__shared__ float upx[N];
	//upper line y
	__shared__ float upy[N];
	//lower like x
	__shared__ float lowx[N];
	//lower line y
	__shared__ float lowy[N];

	//tmp x for batcher's (20 minute path)
	__shared__ float batx[N];
	//tmp y for batcher's
	__shared__ float baty[N];

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


		//set collision as 0
		scol[i] = 0;
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

		//iterate through the drones and compare them to the current drone this thread is handling
		//each thread does an interation on all drones, if collision occures, mark both drones as 1
		for (int q = 0; q < N; q++)
		{
			//make sure we are not comparing the same planes with each other
			if (i != q)
			{
				//for loop to check path up to 20 minutes ahead
				//iterate 20 minutes: 20 x 120 half seconds
				//
				int chk = 0;
				for(int t = 0; t < 20 ; t++)
				{
					//every iteration check for collision, if yes, then go back to beginning, fix angle and check again up to 3 times
					if(chk < 3)
					{
						//increase y bounding box each half second by t * dx
						//x + (t * dx) is how far x will be in t half seconds
						//upper bound
						upx[i] = snx[i] + bound + (t * 120 * dx[i]);
						upy[i] = sny[i] + bound + (t * 120 *dy[i]);
						//lower bound
						lowx[i] = snx[i] - bound + (t * 120 * dx[i]);
						lowy[i] = sny[i] - bound + (t * 120 * dy[i]);

						//collision detection and attempt correction based on angle per half second
						if ((batx[q] <= (upx[i])) && (batx[q] >= (lowx[i])))
						{
							//move plane at angle defined in host function
							tmp[i] = *sn * dx[i];
							tmp0[i] = *cs * dy[i];
							spx[i] = batx[i] + tmp[i];
							spy[i] = baty[i] + tmp0[i];
							batx[i] = spx[i];
							baty[i] = spy[i];
							t = -1;
							chk += 1;
						}
						//detection + correction for y bounding box
						if ((baty[q] <= (upy[i])) && (baty[q] >= (lowy[i])))
						{
							//move plane at angle defined in host function
							tmp[i] = *sn * dx[i];
							tmp0[i] = *cs * dy[i];
							spx[i] = batx[i] + tmp[i];
							spy[i] = baty[i] + tmp0[i];
							batx[i] = spx[i];
							baty[i] = spy[i];
							t = -1;
							chk += 1;
						}
					}

				}

				//check if we corrected less than 3 times and got a good path
				//if so, give the batcher's x and y to shared x and y (with their new rotations)
				if(chk < 3)
				{
					snx[i] = batx[i];
					sny[i] = baty[i];
				}

				//old collision detection and correction not using batchers (path ahead)
				//this just checks for immediate bounding box on current position
				/*
				//collision detection and attempt correction based on angle per half second
				if ((snx[q] <= (snx[i] + bound)) && (snx[q] >= (snx[i] - bound)))
				{
					//move plane at angle defined in host function
					tmp[i] = *sn * dx[i];
					tmp0[i] = *cs * dy[i];
					spx[i] = snx[i] + tmp[i];
					spy[i] = sny[i] + tmp0[i];
					snx[i] = spx[i];
					sny[i] = spy[i];
				}
				//detection + correction for y bounding box
				if ((sny[q] <= (sny[i] + bound)) && (sny[q] >= (sny[i] - bound)))
				{
					//move plane at angle defined in host function
					tmp[i] = *sn * dx[i];
					tmp0[i] = *cs * dy[i];
					spx[i] = snx[i] + tmp[i];
					spy[i] = sny[i] + tmp0[i];
					snx[i] = spx[i];
					sny[i] = spy[i];
				}
				*/

				//now after correction attempt check if still colliding
				//check if drone being compared is within our current drone's 1 nm binding box
				if ((snx[q] <= (snx[i] + bound)) && (snx[q] >= (snx[i] - bound)))
				{
					//check that it hasnt been marked 1 before (collided)
					if ((scol[i] == 0) && (scol[q] == 0))
					{
						scol[i] = 1;
						scol[q] = 1;
					}
					
					
				}
				//same as above for y
				else if ((sny[q] <= (sny[i] + bound)) && (sny[q] >= (sny[i] - bound)))
				{
					if ((scol[i] == 0) && (scol[q] == 0))
					{
						scol[i] = 1;
						scol[q] = 1;
					}
					

				}
				else
				{
					scol[i] = 0;
					scol[q] = 0;
				}
			}
		}

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
	time_t progStart = time(NULL);
	//Drones *drone_h, *drone_d;
	//init random numbers
	//srand(time(NULL));
	//set up host copies
	float *x, *y, *dx, *dy, *nx, *ny, *alt;
	//set up device copies
	float *d_x, *d_y, *d_dx, *d_dy, *d_nx, *d_ny, *d_alt;
	int size = sizeof(float) * N;
	int *col, *d_col;
	float *angle, *theta, *cs, *sn, *d_cs, *d_sn;

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
	nx = (float*)malloc(size);
	ny = (float*)malloc(size);
	alt = (float*)malloc(size);
	col = (int*)malloc(sizeof(int) * N);
	angle = (float*)malloc(size);
	theta = (float*)malloc(size);
	cs = (float*)malloc(size);
	sn = (float*)malloc(size);

	//allocate device memory
	cudaMalloc((void**)&d_x, size);
	cudaMalloc((void**)&d_y, size);
	cudaMalloc((void**)&d_dx, size);
	cudaMalloc((void**)&d_dy, size);
	cudaMalloc((void**)&d_nx, size);
	cudaMalloc((void**)&d_ny, size);
	cudaMalloc((void**)&d_alt, size);
	cudaMalloc((void**)&d_col, sizeof(int)*N);
	cudaMalloc((void**)&d_cs, size);
	cudaMalloc((void**)&d_sn, size);

	//cudaMalloc((void **)&drone_d, sizeof(Drones)*N);
	//drone_h = (Drones *)malloc(sizeof(Drones)*N);
	
	//angle to rotate planes by every half second
	*angle = 3;
	*theta = (*angle)*((pi) / (180.0));
	*cs = cos(*theta);
	*sn = sin(*theta);

	//*dx = ((rand() % 601 + 30) / 3600) * 0.5;
	//*dy = ((rand() % 601 + 30) / 3600) * 0.5;
	
	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, dx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dy, dy, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nx, nx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ny, ny, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_alt, alt, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_col, col, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cs, cs, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sn, sn, size, cudaMemcpyHostToDevice);

	//init drones positions, velocity, and first "next position"
	int blocks = 1;
	setupFlight<<<blocks, N >>>(d_x, d_y, d_dx, d_dy, d_nx, d_ny, d_alt, time(NULL), states);


	cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dx, d_dx, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dy, d_dy, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(nx, d_nx, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(ny, d_ny, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(alt, d_alt, size, cudaMemcpyDeviceToHost);

	printf("Initializing drones...\n");

	for(int j = 0; j < N; j++)
	{
		printf("Drone #:%d x: %f, y: %f, dx: %f, dy: %f, altitude: %f\n", j, x[j], y[j], dx[j], dy[j], alt[j]);
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
		float interval = 0.5;

		time_t start = time(NULL);
		
		//kernel function and memcpy and printing

		moveDrone << <blocks, N >> >(d_x, d_y, d_dx, d_dy, d_col, d_cs, d_sn);

		cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(dx, d_dx, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(dy, d_dy, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(col, d_col, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(cs, d_cs, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(sn, d_sn, size, cudaMemcpyDeviceToHost);

		for (int k = 0; k < N; k++)
		{
			printf("Drone #:%d x: %f, y: %f, dx: %f, dy: %f, col: %d\n", k, x[k], y[k], dx[k], dy[k], col[k]);
			fprintf(file, "%d, %f, %f, %f, %f, %d\n", k, x[k], y[k], dx[k], dy[k], col[k]);
		}

		time_t end = time(NULL);
		count += 1;
		if(count >= duration)
		{
			break;
		}

		float elapsed = difftime(end, start);
		float timeLeft = interval - elapsed;
		printf("time left: %f\n", timeLeft);
		if(timeLeft > 0)
		{
			usleep(timeLeft * 1000 * 1000);
		}
	}

	printf("End flight...\n");

	fclose(file);

	time_t progEnd = time(NULL);
	float totalElapsed = difftime(progEnd, progStart);
	printf("total execution time: %f\n", totalElapsed);

	free(x); free(y); free(dx); free(dy); free(nx); free(ny); free(alt); free(col);

	cudaFree(d_x); cudaFree(d_y); cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_nx); cudaFree(d_ny); cudaFree(d_alt); cudaFree(d_col);

	return 0;
}
