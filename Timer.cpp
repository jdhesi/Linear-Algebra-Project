#include "Timer.h"
#include <stdio.h>
void tic()
{
	startTime = std::chrono::high_resolution_clock::now();
}

void toc()
{
	float dt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000000.f;
	printf("Elapsed time is %f seconds.\n", dt);
}
