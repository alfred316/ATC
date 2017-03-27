
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <windows.h>

#define N 16
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

		//nx[i] = x[i] + dx[i];
		//ny[i] = y[i] + dy[i];

		///*
		//x[i] += dx[i];
		//y[i] += dy[i];
		//*/
		//
		//if (nx[i] >= 128.00)
		//{
		//	nx[i] = nx[i] * (-1.00);
		//}
		//if (ny[i] >= 128.00)
		//{
		//	ny[i] = ny[i] * (-1.00);
		//}

		alt[i] = (curand(&states[threadIdx.x]) % 6001 + 3000);

		col[i] = 0;
		
	}
}


__global__ void TrackDrone(float *x, float *y, float *dx, float *dy, unsigned int seed, curandState_t* states)
{
	int i = threadIdx.x;

	/* we have to initialize the state */
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		threadIdx.x, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[threadIdx.x]);

	float r = curand(&states[threadIdx.x]) / (float)(0x0FFFFFFFFUL);
	float s = curand(&states[threadIdx.x]) / (float)(0x0FFFFFFFFUL);
	float bound = 1;
	int count = 0;
	//vars: hitcount (stores number of times a radar was hit), lasthit (stores id of last radar hit)

	//radar x
	__shared__ float rx[N];
	//radar y
	__shared__ float ry[N];
	//radar hit detection (0, 1, or -1)
	__shared__ int srmatch[N];
	//where we store index of radar hit for each plane
	__shared__ int rmatch[N];

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
	rmatch[i] = -1;

	//create radar by adding random noise to new x and y
	rx[i] = x[i] + r;
	ry[i] = y[i] + s;
	//stores 0, 1 or -1 based on how many planes hit this radar
	srmatch[i] = 0;
	__syncthreads();

	//reverse each quarter of the array
	//this will simulate a random batch of radars coming in that are not corrolated already by thread id

	int quart = (N / 4);

	for(int v = 0; v < 4; ++v)
	{
		int d = quart * (1 + v) - 1;
		int g = quart * v;
		while (g < d)
		{
			float tmpx = rx[g];
			rx[g] = rx[d];
			rx[d] = tmpx;

			float tmpy = ry[g];
			ry[g] = ry[d];
			ry[d] = tmpy; 

			g++;
			d--;
		}
	}

	

	//check against all radars to see if we hit any
	//in here, i is used for the radar index and p is used for the plane index
	//the switch is to make it so each radar iterates in parallel against all planes
	for(int p = 0; p < N; ++p)
	{
		__syncthreads();
		
		//check if new x and y are within 1 nautical mile of radar x and y
		if(x[p] < (rx[i] + bound) && x[p] > (rx[i] - bound))
		{
			//once one is found, change the value of srmatch[p] to 1
			//change value of rmatch[i] to p to store the radar we hit and corrolate it to the plane i
			int set = 0;
			if(srmatch[i] == 0)
			{
				srmatch[i] = 1;
				__syncthreads();
			}
			//if a second one is found change srmatch[p] to -1
			else if(srmatch[i] == 1)
			{
				srmatch[i] = -1;
				__syncthreads();
			}
			
			rmatch[p] = i;
			
		}
		else if (y[p] < (ry[i] + bound) && y[p] > (ry[i] - bound))
		{
			//once one is found, change the value of srmatch[p] to 1
			//change value of rmatch[i] to p to store the radar we hit and corrolate it to the plane i
			
			if (srmatch[i] == 0)
			{
				srmatch[i] = 1;
				__syncthreads();
			}
			//if a second one is found change srmatch[p] to -1
			else if (srmatch[i] == 1)
			{
				srmatch[i] = -1;
				__syncthreads();
			}

			rmatch[p] = i;

		}

	}

	
	//check if there are planes and radars that didnt match, double the bounding box and repeat, up to twice
	//here we go back to having the planes use each thread i index and iterate through the p radars
	if(rmatch[i] == -1)
	{
		while(count < 2)
		{
			//check against all radars to see if we hit any
			for (int p = 0; p < N; ++p)
			{
				__syncthreads();
				
				if(srmatch[p] == 0)
				{
					switch(count)
					{
					case 0:
						bound = bound * 2;
						break;

					case 1:
						bound = bound * 4;
						break;
					}
					//check if new x and y are within 1 nautical mile of radar x and y
					if (x[i] < (rx[p] + bound) && x[i] > (rx[p] - bound))
					{
						//once one is found, change the value of srmatch[p] to 1
						//change value of rmatch[i] to p to store the radar we hit and corrolate it to the plane i

						srmatch[p] = 1;
						__syncthreads();

						rmatch[i] = p;

					}
					else if (y[i] < (ry[p] + bound) && y[i] > (ry[p] - bound))
					{
						//once one is found, change the value of srmatch[p] to 1
						//change value of rmatch[i] to p to store the radar we hit and corrolate it to the plane i

						srmatch[p] = 1;
						__syncthreads();

						rmatch[i] = p;
					}
				}
			}
			count += 1;
		}
	}

	//check our rmatch[i] to see what radar index was hit (where rmatch[i] was not 0)
	//if srmatch[rmatch[i]] == 1, then change new x and y to radar x and y
	//else if srmatch[rmatch[i]] == -1 (hit more than once) or 0 (never hit) then ignore and keep using new x and y
	int m = (int)rmatch[i];
	if(srmatch[m] == 1)
	{
		__syncthreads();
		x[i] = rx[m];
		y[i] = ry[m];
	}

	//otherwise, use new x and y without radar position


}

//move the aircraft by adding its velocity to it's current location
__global__ void CheckCollisionPath(float *x, float *y, float *dx, float *dy, int *col, float *timeTill, float * colWith)
{
	int i = threadIdx.x;
	//create shared variables for next x and y
	//next x
	//__shared__ float snx[N];
	////next y
	//__shared__ float sny[N];
	//collision
	//__shared__ int scol[N];
	//dx
	__shared__ float sdx[N];
	//dy
	__shared__ float sdy[N];
	//temp x for angle calculation
	float spx[N];
	//tmp y for angle calculation
	float spy[N];
	//sin * dx for angle calculation on x axis
	float tmp[N];
	//cos * dy for angle calculation on y axis
	//float tmp0[N];

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
	//time until next collision
	//__shared__ float timeTill[N];
	//id of plane we are colliding with
	//__shared__ float colWith[N];

	float angle[N], theta[N], cs[N], sn[N];

	//calculate shared values
	if(i < N)
	{
		////assign next x and y to shared memory
		//snx[i] = x[i];
		//sny[i] = y[i];
		//init the shared tmp variables
		spx[i] = 0;
		spy[i] = 0;
		tmp[i] = 0;
		tmp[i] = 0;
		//init the upper and lower bounds
		upx[i] = 0;
		upy[i] = 0;
		lowx[i] = 0;
		lowy[i] = 0;

		//init tmp batcher's x and y
		batx[i] = x[i];
		baty[i] = y[i];
		//init shared dx and dy
		sdx[i] = dx[i];
		sdy[i] = dy[i];
		//init timetill and collision with
		timeTill[i] = 300.0;
		colWith[i] = -1;

		__syncthreads();

		//set collision as 0
		//scol[i] = 0;
		//if they go out of bounding grid have it come back around
		/*if (snx[i] >= 128.00)
		{
			snx[i] = snx[i] * (-1.00);
		}
		if (sny[i] >= 128.00)
		{
			sny[i] = sny[i] * (-1.00);
		}*/

		//wait for calculations to be done on all threads
		__syncthreads();

		//iterate through calculated next x and y's for all drones to see if any collisions happen
		//bounding box is 1 nautical mile on each side (so just +-1 on x and y)

		float bound = 1;
		float angleBase = 1;


				//for loop to check path up to 20 minutes ahead
				//iterate 20 minutes: 20 x 120 half seconds
				//
				int chk[N];
				chk[i] = 0;
				int t[N];
				//change to 1 loop starting from 20 ending at more than 20
				for(t[i] = 20; t[i] < 21 ; t[i]++)
				{
					//every iteration check for collision, if yes, then go back to beginning, fix angle and check again up to 3 times
					if(chk[i] < 3)
					{

					//	angle[i] = angleBase + (angleBase * chk[i]);
					//	theta[i] = (angle[i])*((pi) / (180.0));
					//	cs[i] = cos(theta[i]);
					//	sn[i] = sin(theta[i]);

					//	//increase y bounding box each half second by t * dx
					//	//x + (t * dx) is how far x will be in t half seconds
					//	//upper bound
					//	upx[i] = batx[i] + bound + (t[i] * 120 * dx[i]);
					//	upy[i] = baty[i] + bound + (t[i] * 120 *dy[i]);
					//	//lower bound
					//	lowx[i] = batx[i] - bound + (t[i] * 120 * dx[i]);
					//	lowy[i] = baty[i] - bound + (t[i] * 120 * dy[i]);

					//	__syncthreads();

						angle[i] = angleBase + (angleBase * chk[i]);
						theta[i] = (angle[i])*((pi) / (180.0));
						cs[i] = cos(theta[i]);
						sn[i] = sin(theta[i]);

						//get the upper and lower bounds for x and y at the projected time
						//i is our track plane and p is the trial plane

						//track
						//increase y bounding box each half second by t * dx
						//x + (t * dx) is how far x will be in t half seconds
						//upper bound
						upx[i] = batx[i] + bound + (t[i] * 120 * sdx[i]);
						upy[i] = baty[i] + bound + (t[i] * 120 * sdy[i]);
						//lower bound
						lowx[i] = batx[i] - bound + (t[i] * 120 * sdx[i]);
						lowy[i] = baty[i] - bound + (t[i] * 120 * sdy[i]);

						__syncthreads();

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
								
									


									//actual batcher's algorithm code

									float minX, maxX, minY, maxY, timeMin, timeMax;
									float tmpX, tmpY;

									//get min_x
									tmpX = lowx[p] - upx[i];
									minX = fabsf(tmpX);
									minX = minX / (sdx[p] - sdx[i]);
									//get max_x
									tmpX = upx[p] - lowx[i];
									maxX = fabsf(tmpX);
									maxX = maxX / (sdx[p] - sdx[i]);

									//get min_y
									tmpY = lowy[p] - upy[i];
									minY = fabsf(tmpY);
									minY = minY / (sdy[p] - sdy[i]);
									//get max_y
									tmpY = upy[p] - lowy[i];
									maxY = fabsf(tmpY);
									maxY = maxY / (sdy[p] - sdy[i]);

									//get time_min and time_max
									timeMin = max(minX, minY);
									timeMax = min(maxX, maxY);

									//check to see if collision will happen on this path
									if(timeMin < timeMax)
									{
										if (timeMin < timeTill[p])
										{
											//update with the soonest collision time 
											timeTill[i] = timeMin;
											timeTill[p] = timeMin;
											
										}

										//collision
										col[i] = 1;
										col[p] = 1;
										colWith[i] = p;
										colWith[p] = i;

										//change course and see if we're still on a collision course


									}

									

									//either do the resolution here where we figured out there was collision, or do it in a new kernel (might be good option for shared memory sake)

									//this is all wrong vvvv
									
									////collision detection and attempt correction based on angle per half second
									//if ((batx[p] <= (upx[i])) && (batx[p] >= (lowx[i])))
									//{
									//	//move plane at angle defined in host function
									//	/*tmp[i] = *sn * dx[i];
									//	tmp0[i] = *cs * dy[i];
									//	spx[i] = batx[i] + tmp[i];
									//	spy[i] = baty[i] + tmp0[i];
									//	batx[i] = spx[i];
									//	baty[i] = spy[i];*/

									//	batx[i] = cs[i] * snx[i] - sn[i] * sny[i];
									//	baty[i] = sn[i] * snx[i] + cs[i] * sny[i];

									//	t[i] = -1;
									//	chk[i] += 1;
									//	//exit loop
									//	break;
									//}

									//__syncthreads();

									////detection + correction for y bounding box
									//if ((baty[p] <= (upy[i])) && (baty[p] >= (lowy[i])))
									//{
									//	//move plane at angle defined in host function
									//	/*tmp[i] = *sn * dx[i];
									//	tmp0[i] = *cs * dy[i];
									//	spx[i] = batx[i] + tmp[i];
									//	spy[i] = baty[i] + tmp0[i];
									//	batx[i] = spx[i];
									//	baty[i] = spy[i];*/

									//	batx[i] = cs[i] * snx[i] - sn[i] * sny[i];
									//	baty[i] = sn[i] * snx[i] + cs[i] * sny[i];

									//	t[i] = -1;
									//	chk[i] += 1;
									//	//exit loop
									//	break;
									//}

									//__syncthreads();
							}
						}

						////collision detection and attempt correction based on angle per half second
						//if ((batx[q] <= (upx[i])) && (batx[q] >= (lowx[i])))
						//{
						//	//move plane at angle defined in host function
						//	/*tmp[i] = *sn * dx[i];
						//	tmp0[i] = *cs * dy[i];
						//	spx[i] = batx[i] + tmp[i];
						//	spy[i] = baty[i] + tmp0[i];
						//	batx[i] = spx[i];
						//	baty[i] = spy[i];*/

						//	batx[i] = cs[i] * snx[i] - sn[i] * sny[i];
						//	baty[i] = sn[i] *snx[i] + cs[i] *sny[i];

						//	t[i] = -1;
						//	chk[i] += 1;
						//}

						//__syncthreads();

						////detection + correction for y bounding box
						//if ((baty[q] <= (upy[i])) && (baty[q] >= (lowy[i])))
						//{
						//	//move plane at angle defined in host function
						//	/*tmp[i] = *sn * dx[i];
						//	tmp0[i] = *cs * dy[i];
						//	spx[i] = batx[i] + tmp[i];
						//	spy[i] = baty[i] + tmp0[i];
						//	batx[i] = spx[i];
						//	baty[i] = spy[i];*/

						//	batx[i] = cs[i] * snx[i] - sn[i] * sny[i];
						//	baty[i] = sn[i] *snx[i] + cs[i] *sny[i];

						//	t[i] = -1;
						//	chk[i] += 1;
						//}

						//__syncthreads();
					}

				}

				//check if we corrected less than 3 times and got a good path
				//if so, give the batcher's x and y to shared x and y (with their new rotations)
				/*if(chk[i] < 3)
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

				__syncthreads();*/

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

				////now after correction attempt check if still colliding
				////check if drone being compared is within our current drone's 1 nm binding box
				//if ((snx[q] <= (snx[i] + bound)) && (snx[q] >= (snx[i] - bound)))
				//{
				//	//check that it hasnt been marked 1 before (collided)
				//	if ((scol[i] == 0) || (scol[q] == 0))
				//	{
				//		scol[i] = 1;
				//		scol[q] = 1;
				//	}
				//	__syncthreads();
				//	
				//}
				////same as above for y
				//else if ((sny[q] <= (sny[i] + bound)) && (sny[q] >= (sny[i] - bound)))
				//{
				//	if ((scol[i] == 0) || (scol[q] == 0))
				//	{
				//		scol[i] = 1;
				//		scol[q] = 1;
				//	}
				//	
				//	__syncthreads();
				//}
				//else
				//{
				//	scol[i] = scol[i];
				//	scol[q] = scol[q];
				//}
		
		//wait for calculations to be done on all threads
		__syncthreads();

		//give the regular arrays their final values for memcpy by the host
		/*x[i] = snx[i];
		y[i] = sny[i];*/
		//col[i] = scol[i];

	}

	
}


int main(void)
{
	cudaEvent_t allStart, allEnd, setupStart, setupEnd, trackingStart, trackingEnd, eachStepStart, eachStepEnd;
	cudaEventCreate(&allStart);
	cudaEventCreate(&allEnd);
	cudaEventCreate(&setupStart);
	cudaEventCreate(&setupEnd);
	cudaEventCreate(&trackingStart);
	cudaEventCreate(&trackingEnd);
	cudaEventCreate(&eachStepStart);
	cudaEventCreate(&eachStepEnd);

	cudaEventRecord(allStart);

	//time_t progStart = time(NULL);
	//Drones *drone_h, *drone_d;
	//init random numbers
	//srand(time(NULL));
	//set up host copies
	float *x, *y, *dx, *dy, *alt;
	//set up device copies
	float *d_x, *d_y, *d_dx, *d_dy, *d_alt;
	int size = sizeof(float) * N;
	int *col, *d_col;
	float *angle, *theta, *cs, *sn, *d_cs, *d_sn;
	float *h_timeTill, *h_colWith; // *h_upx, *h_upy, *h_lowx, *h_lowy;
	float *d_timeTill, *d_colWith; // *d_upx, *d_upy, *d_lowx, *d_lowy;
	
	
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
	angle = (float*)malloc(size);
	theta = (float*)malloc(size);
	cs = (float*)malloc(size);
	sn = (float*)malloc(size);
	h_timeTill = (float*)malloc(size);
	h_colWith = (float*)malloc(size);
	/*h_upx = (float*)malloc(size);
	h_upy = (float*)malloc(size);
	h_lowx = (float*)malloc(size);
	h_lowy = (float*)malloc(size);*/

	//allocate device memory
	cudaMalloc((void**)&d_x, size);
	cudaMalloc((void**)&d_y, size);
	cudaMalloc((void**)&d_dx, size);
	cudaMalloc((void**)&d_dy, size);
	cudaMalloc((void**)&d_alt, size);
	cudaMalloc((void**)&d_col, sizeof(int)*N);
	cudaMalloc((void**)&d_cs, size);
	cudaMalloc((void**)&d_sn, size);
	cudaMalloc((void**)&d_timeTill, size);
	cudaMalloc((void**)&d_colWith, size);
	/*cudaMalloc((void**)&d_upx, size);
	cudaMalloc((void**)&d_upy, size);
	cudaMalloc((void**)&d_lowx, size);
	cudaMalloc((void**)&d_lowy, size);*/

	//cudaMalloc((void **)&drone_d, sizeof(Drones)*N);
	//drone_h = (Drones *)malloc(sizeof(Drones)*N);
	
	//angle to rotate planes by every half second
	/**angle = 3;
	*theta = (*angle)*((pi) / (180.0));
	*cs = cos(*theta);
	*sn = sin(*theta);*/

	//*dx = ((rand() % 601 + 30) / 3600) * 0.5;
	//*dy = ((rand() % 601 + 30) / 3600) * 0.5;
	
	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, dx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dy, dy, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_alt, alt, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_col, col, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cs, cs, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sn, sn, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_timeTill, h_timeTill, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_colWith, h_colWith, size, cudaMemcpyHostToDevice);
	/*cudaMemcpy(d_upx, h_upx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_upy, h_upy, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lowx, h_lowx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lowy, d_lowy, size, cudaMemcpyHostToDevice);*/

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
	cudaMemcpy(alt, d_alt, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(col, d_col, size, cudaMemcpyDeviceToHost);

	printf("Initializing drones...\n");

	/*for(int j = 0; j < N; j++)
	{
		printf("Drone #:%d x: %f, y: %f, dx: %f, dy: %f, altitude: %f\n", j, x[j], y[j], dx[j], dy[j], alt[j]);
	}*/
	
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

		cudaEventRecord(trackingStart);
		//implement tracking with radar in it's own kernel function before collision detection and resolution
		TrackDrone << <blocks, N >> > (d_x, d_y, d_dx, d_dy, time(NULL), states);
		cudaEventRecord(trackingEnd);
		cudaEventSynchronize(trackingEnd);

		float trackingTime;
		cudaEventElapsedTime(&trackingTime, trackingStart, trackingEnd);
		printf("Each Iteration of Flights Tracking Time for %d drones: %f ms\n", N, trackingTime);


		cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

		//time_t start = time(NULL);

		//kernel function and memcpy and printing

		cudaEventRecord(eachStepStart);
		//add terrain avoidance kernel too somewhere

		//only do collision detection and resolution every few seconds and not every half second step
		CheckCollisionPath << <blocks, N >> >(d_x, d_y, d_dx, d_dy, d_col, d_timeTill, d_colWith);

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
		cudaMemcpy(cs, d_cs, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(sn, d_sn, size, cudaMemcpyDeviceToHost);

		
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
			//sleep works with ms
			Sleep(timeLeft);
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

	cudaEventDestroy(allStart);
	cudaEventDestroy(allEnd);
	cudaEventDestroy(setupStart);
	cudaEventDestroy(setupEnd);
	cudaEventDestroy(eachStepStart);
	cudaEventDestroy(eachStepEnd);

	free(x); free(y); free(dx); free(dy); free(alt); free(col);

	cudaFree(d_x); cudaFree(d_y); cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_alt); cudaFree(d_col);

	return 0;
}

