#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512

struct Drone {
	float x;
	float y;
	float dx;
	float dy;
}

__global__ void flight(float x, float y, float dx, float dy, int n)
{
	x = rand() % 129;
	y = rand() % 129;
	
	//drone.dx = ((rand() % 600 + 30)/3600) * 0.5;	

	int i = 0;	
	if(i < n)
	{
		drone.x += dx;
		drone.y += dy;
		if(drone.x >= 128)
		{
			drone.x = drone.x * (-1);
		}		
		if(drone.y >= 128)
		{
			drone.y = drone.y * (-1);
		}
		printf("", drone.x, drone.y);

	}
}


int main (void)
{
	srand(time(NULL));
	
	//initialize drone
	struct Drone drone;
	//init random value for dx and dy	
	drone.dx = ((rand() % 600 + 30)/3600) * 0.5;
	drone.dy = ((rand() % 600 + 30)/3600) * 0.5;
		
	
	//set up device copies
	float *d_x, *d_y, *d_dx, *d_dy;
	int size = sizeof(float);

	cudaMalloc((void **), &d_x, size);
	cudaMalloc((void **), &d_y, size);
	cudaMalloc((void **), &d_dx, size);
	cudaMalloc((void **), &d_dy, size);
 
}
