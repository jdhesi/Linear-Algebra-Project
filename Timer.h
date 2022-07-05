#ifndef TIMERDEF
#define TIMERDEF

#include <chrono>

static std::chrono::
time_point<std::chrono::high_resolution_clock> startTime;

void tic();
void toc();

#endif // !TIMERDEF