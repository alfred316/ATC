#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM 512

struct drone {
	float x;
	float y;
	float dx;
	float dy;
}

__global__ void flight(struct drone, int n)
{
	drone.x = rand() % 129;
	drone.y = rand() % 129;
	
	//drone.dx = ((rand() % 600 + 30)/3600) * 0.5;	

	int i = 0;	
	while (i < n)
	{
		drone.dx = ((rand() % 600 + 30)/3600) * 0.5;
		drone.dy = ((rand() % 600 + 30)/3600) * 0.5;
		drone.x += drone.dx;
		drone.y += drone.dy;
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

}
