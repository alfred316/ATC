/*

Alfred Shaker
Air Traffic Control CUDA program
Latest Update: April 1st 2017

*/
/*LIMIT BREAK METHOD*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <unistd.h>

#define N 32640 //needs to be a multiple of 128 (block size) //32640
#define pi 3.14159265358979323846


struct drone
{
float *x; //x coordinate
float *y; //y coordinate
float *batx;//for collision correction batcher's alg
float *baty;//for collision correction batcher's alg
float *dx; //speed on x
float *dy; //speed on y
float *alt; //altitude
int *col; //if on collision course
float *timeTill; //shorted time until next collision
int *colWith; //drone we will collide with
int *rMatchWith; //what radar matched with
int *rMatch;
} drones[N];

struct radar
{
	float *rx;
	float *ry;
	int *rMatch; //0, 1 or -1 based on not hit, hit, or hit too many times
	int *rMatchWith;
} radars[N];


//__constant__ float *d_x, *d_y, *d_dx, *d_dy, *d_nx, *d_ny;

//init the positions and velocities of the drones and also calculate their first next position
__global__ void setupFlight(float *x, float *y, float *dx, float *dy, float *alt, int *col, unsigned int seed, curandState_t* states)
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;


	if (i < N)
	{
		/* we have to initialize the state */
		curand_init(seed, /* the seed controls the sequence of random values that are produced */
			i, /* the sequence number is only important with multiple cores */
			0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
			&states[i]);

		//calculate initial pos and velocity
		x[i] = curand(&states[i]) % 129;
		y[i] = curand(&states[i]) % 129;

		int tmpx = curand(&states[i]) % 50;
		if (tmpx % 2 == 0)
		{
			x[i] = -1 * x[i];
		}
		int tmpy = curand(&states[i]) % 50;
		if (tmpy % 2 != 0)
		{
			y[i] = -1 * y[i];
		}

		float S = 0.f;
		//calculate random speed
		dx[i] = (curand(&states[i]) % 601 + 30);
		dy[i] = (curand(&states[i]) % 601 + 30);

		S = (curand(&states[i]) % 601 + 30);

		float tmpVel = pow(S, 2) - pow(dx[i], 2);
		tmpVel = fabsf(tmpVel);
		dy[i] = sqrt(tmpVel);

		//dy[i] = fabsf(dy[i]);

		//knots per half second
		dx[i] = (dx[i] / 3600) * 0.5;
		dy[i] = (dy[i] / 3600) * 0.5;

		int tmpdx = curand(&states[i]) % 50;
		if (tmpdx % 2 == 0)
		{
			dx[i] = -1 * dx[i];
		}
		int tmpdy = curand(&states[i]) % 50;
		if (tmpdy % 2 != 0)
		{
			dy[i] = -1 * dy[i];
		}

		alt[i] = (curand(&states[i]) % 6001 + 500);

		col[i] = 0;

	}
}

__global__ void GenerateRadarData(float *x, float *y, float *dx, float *dy, float *rx, float *ry, int *rMatch, unsigned int seed, curandState_t* states)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	/* we have to initialize the state */
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		i, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[i]);

	float r = curand(&states[i]) / (float)(0x0FFFFFFFFUL);
	float s = curand(&states[i]) / (float)(0x0FFFFFFFFUL);


	//create radar by adding random noise to new x and y
	rx[i] = x[i] + dx[i] + r;
	ry[i] = y[i] + dy[i] + s;

	int tmps = curand(&states[i]) % 50;
	if (tmps % 2 == 0)
	{
		s = -1 * s;
	}
	int tmpr = curand(&states[i]) % 50;
	if (tmpr % 2 != 0)
	{
		r = -1 * r;
	}


	//create radar by adding random noise to new x and y
	rx[i] = x[i] + dx[i] + r;
	ry[i] = y[i] + dy[i] + s;

	if (rx[i] >= 128.00)
	{
		rx[i] = rx[i] * (-1.00);
	}
	if (ry[i] >= 128.00)
	{
		ry[i] = ry[i] * (-1.00);
	}

	//stores 0, 1 or -1 based on how many planes hit this radar
	rMatch[i] = 0;
}

__global__ void TrackDrone(float *x, float *y, float *dx, float *dy, float *rx, float *ry, int *rMatch, int *rMatchWith)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i < N)
	{
	
		float bound = 0.5;
		int count = 0;

		//move planes 
		x[i] = x[i] + dx[i];
		y[i] = y[i] + dy[i];

		//check boundaries
		if (x[i] >= 128.00)
		{
			x[i] = x[i] * (-1.00);
		}
		if (y[i] >= 128.00)
		{
			y[i] = y[i] * (-1.00);
		}

		//stores the index of the radar corrolated
		//radar
		rMatchWith[i] = -1;
		//drones
		rMatch[i] = 0;
		//_syncthreads();


		//check against all radars to see if we hit any
		for (int p = 0; p < N; p++)
		{
			
			//check if new x and y are within 1 nautical mile of radar x and y
			if ((rx[i] < (x[p] + bound) && rx[i] > (x[p] - bound)) && (ry[i] < (y[p] + bound) && ry[i] > (y[p] - bound)))
			{
				//once one is found, change the value of srmatch[p] to 1
				//change value of rmatch[i] to p to store the radar we hit and corrolate it to the plane i
				//int set = 0;
				if (rMatch[p] == 0)
				{
					rMatch[p] = 1;

				}
				//if a second radar matches this drone set drone rmatch to -1 to discard drone matching
				else if (rMatch[p] == 1)
				{
					rMatch[p] = -1;

				}

				//first aircraft for radar to match with
				if(rMatchWith[i] == -1)
				{
					rMatchWith[i] = p;
				}
				//radar has matched an aircraft before, must be discarded for matching multiple aircrafts
				else
				{
					int tmp = rMatchWith[i]; //find previously matched drone
					rMatch[tmp] = 0; //unmatch previously matched drone
					rMatch[p] = 0;
					rMatchWith[i] = -2; //-2 means we are discarding it
					break;
				}

				
				//_syncthreads();
			}
		}

		
		int doneMatching = 0;

		//check if there are planes and radars that didnt match, double the bounding box and repeat, up to twice
		//here we go back to having the planes use each thread i index and iterate through the p radars
		if (rMatchWith[i] == -1) //-1 means not matched yet. 1 means matched, -2 means discarded
		{
			while (count < 2)
			{
				switch (count)
				{
				case 0:
					bound = bound * 2;
					break;

				case 1:
					bound = bound * 4;
					break;
				}
				//check against all drones 
				for (int p = 0; p < N; p++)
				{
					

					if (rMatch[p] == 0) //drone hasnt been matched once
					{

						//check if radar is within drone bounds
						if ((rx[i] < (x[p] + bound) && rx[i] > (x[p] - bound)) && (ry[i] < (y[p] + bound) && ry[i] > (y[p] - bound)))
						{
							//once one is found, change the value of srmatch[p] to 1
							//change value of rmatch[i] to p to store the radar we hit and corrolate it to the plane i

							rMatch[p] = 1;

							//first aircraft for radar to match with
							if (rMatchWith[i] == -1)
							{
								rMatchWith[i] = p;
								doneMatching = 1;
							}
							//radar has matched an aircraft before, must be discarded for matching multiple aircrafts
							else
							{
								int tmp = rMatchWith[i]; //find previously matched drone
								rMatch[tmp] = -1; //unmatch previously matched drone
								rMatchWith[i] = -2; //-2 means we are discarding it
								break;
							}

						}
					}

					if (doneMatching == 1)
						break; //if matched, we dont need to check anymore
				}
				if (doneMatching == 0)
				{
					count += 1;
				}
				else if (doneMatching == 1)
					break; //if matched, we dont need to check anymore
				
			}
		}

		//check our rMatchWith[i] to see what radar index was hit (where srmatch[i] was not 0)
		//if srmatch[rmatch[i]] == 1, then change new x and y to radar x and y
		//else if srmatch[rmatch[i]] == -1 (hit more than once) or 0 (never hit) then ignore and keep using new x and y
		int m = (int)rMatchWith[i];
		if (m != -1 && m != -2)
		{
			if (rMatch[m] == 1)
			{
				x[m] = rx[i];
				y[m] = ry[i];
				//_syncthreads();
			}
		}

		
		//otherwise, use new x and y without radar position

	}
	//_syncthreads();
}

//move the aircraft by adding its velocity to it's current location
__global__ void CheckCollisionPath(float *x, float *y, float *batx, float *baty, float *dx, float *dy, int *col, float *timeTill, int * colWith, float * alt)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	int angle;
	float theta, cs, sn;

	//calculate shared values
	if (i < N)
	{
		
		//init timetill and collision with
		timeTill[i] = 300.0;
		colWith[i] = -1;
		col[i] = 0;
		batx[i] = x[i];
		baty[i] = y[i];
		//wait for calculations to be done on all threads
		//_syncthreads();

		//iterate through calculated next x and y's for all drones to see if any collisions happen
		//bounding box is 1 nautical mile on each side (so just +-1 on x and y)

		//float bound = 1.5; //in nautical miles
		float angleBase = 5; //in degrees


							 //for loop to check path up to 20 minutes ahead
							 //iterate 20 minutes: 20 x 120 half seconds

		int chk;
		chk = 0;
		int t;
		//change to 1 loop starting from 20 ending at more than 20
		for (t = 20; t < 21; t++)
		{
			//every iteration check for collision, if yes, then go back to beginning, fix angle and check again up to 3 times
			for (int p = 0; p < N; p++)
			{

				

				//get the upper and lower bounds for x and y at the projected time
				//i is our track plane and p is the trial plane


				//may or may not have to correct these paths to go back around if out of bounds. 


				//loop all Q planes and check if any other plane is in our bounding box
				//might be time consuming but will work
				//this is instead of checking against 1 plane 20 times for each device

				//iterate through the drones and compare them to the current drone this thread is handling
				//each thread does an interation on all drones, if collision occures, mark both drones as 1
				if (i != p && p < N && (alt[p] < (alt[i] + 500) && alt[p] > (alt[i] - 500)))
				{

					float minX, maxX, minY, maxY, timeMin, timeMax;
						float tmpX, tmpY;



						//get min_x
						tmpX = (x[p] +(t * 120 * dx[i]) )- (batx[i] + (t * 120 * dx[i]));
						minX = fabsf(tmpX);
						minX = minX - 3;
						minX = minX / fabsf(dx[p] - dx[i]);
						//get max_x
						tmpX = (x[p] + (t * 120 * dx[i])) - (batx[i] + (t * 120 * dx[i]));
						maxX = fabsf(tmpX);
						maxX = maxX + 3;
						maxX = maxX / fabsf(dx[p] - dx[i]);

						//get min_y
						tmpY = (y[p] + (t * 120 * dy[i])) - (baty[i] + (t * 120 * dy[i]));
						minY = fabsf(tmpY);
						minY = minY - 3;
						minY = minY / fabsf(dy[p] - dy[i]);
						//get max_y
						tmpY = (y[p] + (t * 120 * dy[i])) - (baty[i] + (t * 120 * dy[i]));
						maxY = fabsf(tmpY);
						maxY = maxY + 3;
						maxY = maxY / fabsf(dy[p] - dy[i]);

						//get time_min and time_max
						timeMin = max(minX, minY);
						timeMax = min(maxX, maxY);

					//make sure we are not comparing the same planes with each other and that the planes are within 1000 nautical miles of each other
					if (timeMin < timeMax)
					{

						if (timeMin < timeTill[i])
							{

								chk++;

								//update with the soonest collision time 
								timeTill[i] = timeMin;

								angle = angleBase;

								if(angle < 30)
								{

									if(angle > 0)
									{
										angle = (-1) * angle;
									}
									else if(angle < 0)
									{
										angle = (-1) * angle;
										angle += 5;
									}
								}
								theta = (angle)*((pi) / (180.0));
								cs = cos(theta);
								sn = sin(theta);


								//set collision variables collision
								col[i] = 1;
								col[p] = 1;
								colWith[i] = p;
								colWith[p] = i;
								//_syncthreads();
								//change course and see if we're still on a collision course

								//move x and y by a certain angle 
								batx[i] = cs * x[i] - sn * y[i];
								baty[i] = sn * x[i] + cs * y[i];
								//_syncthreads();
								/*batx[p] = cs * x[p] + sn * y[p];
								baty[p] = sn * x[p] + cs * y[p];
								//_syncthreads();*/
								//reset timer and increment check and try again by calling a break out of p loop 
								//and reset t loop back to 20 so repeat the process with new batx and baty and increment check
								t = 19;
								//chk += 1;
								break;
							}
						
						
					}
				}

				//check if no collision but course corrected AND we have finished checking all planes to assign new x and y based on course correction
						if (chk > 0 && p == (N - 1))
						{
							x[i] = batx[i];
							y[i] = baty[i];
							//_syncthreads();
							col[i] = 0;
							col[p] = 0;
							colWith[i] = -1;
							colWith[p] = -1;
							//_syncthreads();
						}
						//else if no collision and no course correction then do nothing to x and y

			}

		}


	}


}


int main(void)
{
	cudaEvent_t allStart, allEnd, setupStart, setupEnd, trackingStart, trackingEnd, CollisionDetectionStart, CollisionDetectionEnd;
	cudaEventCreate(&allStart);
	cudaEventCreate(&allEnd);
	cudaEventCreate(&setupStart);
	cudaEventCreate(&setupEnd);
	cudaEventCreate(&trackingStart);
	cudaEventCreate(&trackingEnd);
	cudaEventCreate(&CollisionDetectionStart);
	cudaEventCreate(&CollisionDetectionEnd);

	cudaEventRecord(allStart);

	//time_t progStart = time(NULL);
	//Drones *drone_h, *drone_d;
	//init random numbers
	//srand(time(NULL));
	//set up host copies
	float *x, *y, *batx, *baty, *dx, *dy, *alt, *rx, *ry;
	//set up device copies
	//float *d_x, *d_y, *d_dx, *d_dy, *d_alt;
	int size = sizeof(float) * N;
	int *col, *rMatch, *rMatchWith;
	float *cs, *sn, *d_cs, *d_sn;
	float *h_timeTill, *h_colWith; // *h_upx, *h_upy, *h_lowx, *h_lowy;
	//float *d_timeTill, *d_colWith; // *d_upx, *d_upy, *d_lowx, *d_lowy;


								   /* CUDA's random number library uses curandState_t to keep track of the seed value
								   we will store a random state for every thread  */
	curandState_t* states;

	/* allocate space on the GPU for the random states */
	cudaMalloc((void**)&states, N * sizeof(curandState_t));

	//file
	/*FILE *file;
	char *fileName = "drones.csv";
	file = fopen(fileName, "w+");
	fprintf(file, "Drone Id, X, Y, dX, dY, Collision\n");
	*/
	//drone *Drones = new drone;

	//allocate host memory
	x = (float*)malloc(size);
	y = (float*)malloc(size);
	dx = (float*)malloc(size);
	dy = (float*)malloc(size);
	alt = (float*)malloc(size);
	col = (int*)malloc(sizeof(int) * N);
	//angle = (float*)malloc(size);
	//theta = (float*)malloc(size);
	cs = (float*)malloc(size);
	sn = (float*)malloc(size);
	h_timeTill = (float*)malloc(size);
	h_colWith = (float*)malloc(size);
	rx = (float*)malloc(size);
	ry = (float *)malloc(size);
	rMatch = (int *)malloc(sizeof(int) * N);
	rMatchWith = (int *)malloc(sizeof(int) * N);
	batx = (float*)malloc(size);
	baty = (float*)malloc(size);
	/*h_upx = (float*)malloc(size);
	h_upy = (float*)malloc(size);
	h_lowx = (float*)malloc(size);
	h_lowy = (float*)malloc(size);*/

	//allocate device memory
	cudaMalloc((void**)&drones->x, size);
	cudaMalloc((void**)&drones->y, size);
	cudaMalloc((void**)&drones->batx, size);
	cudaMalloc((void**)&drones->baty, size);
	cudaMalloc((void**)&drones->dx, size);
	cudaMalloc((void**)&drones->dy, size);
	cudaMalloc((void**)&drones->alt, size);
	cudaMalloc((void**)&drones->col, sizeof(int)*N);
	cudaMalloc((void**)&d_cs, size);
	cudaMalloc((void**)&d_sn, size);
	cudaMalloc((void**)&drones->timeTill, size);
	cudaMalloc((void**)&drones->colWith, size);
	cudaMalloc((void**)&radars->rx, size);
	cudaMalloc((void**)&radars->ry, size);
	cudaMalloc((void**)&radars->rMatch, sizeof(int)*N);
	cudaMalloc((void**)&drones->rMatchWith, sizeof(int)*N);
	cudaMalloc((void**)&drones->rMatch, sizeof(int)*N);
	cudaMalloc((void**)&radars->rMatchWith, sizeof(int)*N);
	/*cudaMalloc((void**)&d_upx, size);
	cudaMalloc((void**)&d_upy, size);
	cudaMalloc((void**)&d_lowx, size);
	cudaMalloc((void**)&d_lowy, size);*/

	//cudaMalloc((void **)&drone_d, sizeof(Drones)*N);
	//drone_h = (Drones *)malloc(sizeof(Drones)*N);
	
	
	cudaMemcpy(drones->x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->y, y, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->batx, batx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->baty, baty, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->dx, dx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->dy, dy, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->alt, alt, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->col, col, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cs, cs, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sn, sn, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->timeTill, h_timeTill, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->colWith, h_colWith, size, cudaMemcpyHostToDevice);
	cudaMemcpy(radars->rx, rx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(radars->ry, ry, size, cudaMemcpyHostToDevice);
	cudaMemcpy(radars->rMatch, rMatch, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->rMatchWith, rMatchWith, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(drones->rMatch, rMatch, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(radars->rMatchWith, rMatchWith, sizeof(int)*N, cudaMemcpyHostToDevice);
	/*cudaMemcpy(d_upx, h_upx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_upy, h_upy, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lowx, h_lowx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lowy, d_lowy, size, cudaMemcpyHostToDevice);*/

	//init drones positions, velocity, and first "next position"
	int blocks = 0;
	int threads = 0;
	if (N < 96)
	{
		blocks = (N + (N - 1)) / N;
		threads = N;
	}
	else
	{
		blocks = (N + 95) / 96;
		threads = 96;
	}

	cudaEventRecord(setupStart);

	setupFlight << <blocks, threads >> >(drones->x, drones->y, drones->dx, drones->dy, drones->alt, drones->col, time(NULL), states);

	cudaEventRecord(setupEnd);
	cudaEventSynchronize(setupEnd);
	float setupTime;
	cudaEventElapsedTime(&setupTime, setupStart, setupEnd);
	printf("Setup Flights Time for %d drones: %f ms\n", N, setupTime);


	cudaMemcpy(x, drones->x, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, drones->y, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dx, drones->dx, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dy, drones->dy, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(alt, drones->alt, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(col, drones->col, size, cudaMemcpyDeviceToHost);

	printf("Initializing drones...\n");

	/*for(int j = 0; j < N; j++)
	{
	printf("Drone #:%d x: %f, y: %f, dx: %f, dy: %f, altitude: %f\n", j, x[j], y[j], dx[j], dy[j], alt[j]);
	}*/

	//radar and collision detection
	//new kernel function to take all these values and keep adding their dx/dy and 
	//checking nx/ny with 1 nautical mile bounding box (+1/-1 on x and y from current x and y)


	/*GENERATE INITIAL RADAR REPORTS*/
	/*RADAR INIT KERNEL*/
	GenerateRadarData << <blocks, threads >> > (drones->x, drones->y, drones->dx, drones->dy, radars->rx, radars->ry, drones->rMatch, time(NULL), states);

	cudaMemcpy(rx, radars->rx, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(ry, radars->ry, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(rMatch, radars->rMatch, size, cudaMemcpyDeviceToHost);

	//reverse each quarter of the array
	//this will simulate a random batch of radars coming in that are not corrolated already by thread id


	int quart = (N / 4);
	float tmpx = 0;
	float tmpy = 0;
	int d = 0;
	int g = 0;
	int v = 0;
	for (v = 0; v < 4; v++)
	{
		d = quart * (1 + v) - 1;
		g = quart * v;
		while (g < d)
		{
			tmpx = rx[g];
			rx[g] = rx[d];
			rx[d] = tmpx;

			tmpy = ry[g];
			ry[g] = ry[d];
			ry[d] = tmpy;

			g++;
			d--;
		}
	}

	printf("Flying drones...\n");

	//each count represents half a second
	int count = 0;
	//duration is how many seconds we want to test this for multiplied by 2
	int duration = 64;

	float trackTime[duration];
	float cdrTime[4];

	//loop infinitely until the duration condition is met
	//each iteration run the moveDrone kernel function to move the drone and check collision
	//make sure each iteration takes half a second
	for (;;)
	{
		//interval of half a second, which is 500ms
		float interval = 500.f;

		cudaMemcpy(radars->rx, rx, size, cudaMemcpyHostToDevice);
		cudaMemcpy(radars->ry, ry, size, cudaMemcpyHostToDevice);
		cudaMemcpy(radars->rMatch, rMatch, sizeof(int)*N, cudaMemcpyHostToDevice);
		cudaEventRecord(trackingStart);
		//implement tracking with radar in it's own kernel function before collision detection and resolution
		TrackDrone << <blocks, threads >> > (drones->x, drones->y, drones->dx, drones->dy, radars->rx, radars->ry, drones->rMatch, radars->rMatchWith);
		cudaEventRecord(trackingEnd);
		cudaEventSynchronize(trackingEnd);

		float trackingTime;
		cudaEventElapsedTime(&trackingTime, trackingStart, trackingEnd);
		printf("Each Iteration of Flights Tracking Time for %d drones: %f ms\n", N, trackingTime);

		trackTime[count] = trackingTime;

		cudaMemcpy(x, drones->x, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(y, drones->y, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(rx, radars->rx, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(ry, radars->ry, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(rMatch, drones->rMatch, sizeof(int)*N, cudaMemcpyDeviceToHost);
		cudaMemcpy(rMatchWith, radars->rMatchWith, sizeof(int)*N, cudaMemcpyDeviceToHost);

		/*for (int k = 0; k < N; k++)
		{
			printf("Drone #:%d x: %f, y: %f, rx: %f, ry: %f, rMatch: %d, rMatchWith: %d\n", k, x[k], y[k], rx[k], ry[k], rMatch[k], rMatchWith[k]);
		}*/

		//time_t start = time(NULL);

		//kernel function and memcpy and printing

		//Collision detection & resolution
		//happens only three times during entire duration

		//int half_dur = duration / 2;
		int almostDone = duration - 1;
		float CollisionDetectionTime = 0;

		if (count == 15 || count == 31 || count == 47 || count == 63)
		{

			/*cudaMemcpy(drones->x, x, size, cudaMemcpyHostToDevice);
			cudaMemcpy(drones->y, y, size, cudaMemcpyHostToDevice);*/

			cudaEventRecord(CollisionDetectionStart);
			//add terrain avoidance kernel too somewhere

			//only do collision detection and resolution every few seconds and not every half second step
			CheckCollisionPath << <blocks, threads >> >(drones->x, drones->y, drones->batx, drones->baty, drones->dx, drones->dy, drones->col, drones->timeTill, drones->colWith, drones->alt);

			cudaEventRecord(CollisionDetectionEnd);
			cudaEventSynchronize(CollisionDetectionEnd);

			cudaEventElapsedTime(&CollisionDetectionTime, CollisionDetectionStart, CollisionDetectionEnd);
			printf("Collision detection & resolution for %d drones: %f ms\n", N, CollisionDetectionTime);

			switch(count)
			{
			case 15:
				cdrTime[0] = CollisionDetectionTime;
				break;
			case 31:
				cdrTime[1] = CollisionDetectionTime;
				break;
			case 47:
				cdrTime[2] = CollisionDetectionTime;
				break;
			case 63:
				cdrTime[3] = CollisionDetectionTime;
				break;
			}

			cudaMemcpy(x, drones->x, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(y, drones->y, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(dx, drones->dx, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(dy, drones->dy, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(col, drones->col, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_timeTill, drones->timeTill, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_colWith, drones->colWith, size, cudaMemcpyDeviceToHost);

			//for (int k = 0; k < N; k++)
			//{
			//printf("Drone #:%d x: %f, y: %f, dx: %f, dy: %f, col: %d\n", k, x[k], y[k], dx[k], dy[k], col[k]);
			//fprintf(file, "%d, %f, %f, %f, %f, %d\n", k, x[k], y[k], dx[k], dy[k], col[k]);
			//}
		}

		/*GENERATE NEW RADAR REPORTS*/
		GenerateRadarData << <blocks, threads >> > (drones->x, drones->y, drones->dx, drones->dy, radars->rx, radars->ry, drones->rMatch, time(NULL), states);

		cudaMemcpy(rx, radars->rx, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(ry, radars->ry, size, cudaMemcpyDeviceToHost);

		//reverse each quarter of the array
		//this will simulate a random batch of radars coming in that are not corrolated already by thread id
		//shuffle that shit

		quart = (N / 4);
		tmpx = 0;
		tmpy = 0;
		d = 0;
		g = 0;
		v = 0;
		for (v = 0; v < 4; v++)
		{
			d = quart * (1 + v) - 1;
			g = quart * v;
			while (g < d)
			{
				tmpx = rx[g];
				rx[g] = rx[d];
				rx[d] = tmpx;

				tmpy = ry[g];
				ry[g] = ry[d];
				ry[d] = tmpy;

				g++;
				d--;
			}
		}


		//time_t end = time(NULL);

		count += 1;
		if (count >= duration)
		{
			break;
		}

		//if execution done in under half second, wait the rest of the time to avoid moving
		//more than once every half second
		//elapsed and timeleft are in ms (milliseconds)
		float elapsed = CollisionDetectionTime + trackingTime;
		float timeLeft = interval - elapsed;
		printf("time left: %f\n", timeLeft);
		//makes sure we dont take more than 500ms to execute
		if (timeLeft > 0)
		{
			//sleep works with ms
			usleep(timeLeft * 1000);
		}
	}

	printf("End flight...\n");

	cudaEventRecord(allEnd);
	cudaEventSynchronize(allEnd);
	float allTime;
	cudaEventElapsedTime(&allTime, allStart, allEnd);
	printf("Total Execution Time for %d flights: %f ms\n", N, allTime);

	//fclose(file);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	//printf("Name: %s \n", prop.name);
	char *deviceName = prop.name;
	char *fext = ".csv";
	float strsize = strlen(deviceName) + strlen(fext);
	//file
	FILE *file;
	char* fileName;
	fileName = (char *)malloc(strsize);
	strcpy(fileName, deviceName);
	strcat(fileName, fext);
	file = fopen(fileName, "w+");
	if(N == 96)
	{
		fprintf(file, "AircraftCount, Track, CDR, Track+CDR\n");
	}
	

	float sum = 0;
	for(int h = 0; h < duration; h++)
	{
		sum += trackTime[h];
	}

	float avgTrack = sum / duration;
	printf("\naverage execution time for tracking and correlation task: %f ms\n", avgTrack);

	float cdrSum = 0;
	for(int l = 0; l < 4; l++)
	{
		cdrSum += cdrTime[l];
	}

	float avgCdr = cdrSum / 4;
	printf("average execution time for collision detection and resolution task: %f ms\n", avgCdr);

	float allTaskAvg = avgCdr + avgTrack;
	printf("average execution time for both tasks combined per cycle task: %f ms\n", allTaskAvg);
	fprintf(file, "%d, %f, %f, %f\n", N, avgTrack, avgCdr, allTaskAvg);

	fclose(file);

	/*
	time_t progEnd = time(NULL);
	float totalElapsed = difftime(progEnd, progStart);
	printf("total execution time: %f\n", totalElapsed);
	*/

	cudaEventDestroy(allStart);
	cudaEventDestroy(allEnd);
	cudaEventDestroy(setupStart);
	cudaEventDestroy(setupEnd);
	cudaEventDestroy(CollisionDetectionStart);
	cudaEventDestroy(CollisionDetectionEnd);

	free(x); free(y); free(dx); free(dy); free(alt); free(col);

	cudaFree(drones->x); cudaFree(drones->y); cudaFree(drones->dx); cudaFree(drones->dy); cudaFree(drones->alt); cudaFree(drones->col);

	return 0;
}
