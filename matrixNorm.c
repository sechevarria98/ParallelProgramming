#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define N 6000

/* Matrices */
volatile float A[N][N], B[N][N];


/* Initialize A and B*/
void initialize_inputs() {
    int row, col;
    
    srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row][col] = (float)rand() / 32768.0;
            B[row][col] = 0.0;
        }
    } 
}

/* Kernel function */

void matrixNorm() {
    int row, col;
    float mu, sigma;
    
    printf("Computing Serially.\n");
    
    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row][col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                B[row][col] = 0.0;
            else
                B[row][col] = (A[row][col] - mu) / sigma;
        }
    }
    
}



int main(int argc, char **argv) {
    struct timeval start, stop;
    struct timezone tzdummy;
    unsigned long long runtime;
    
    initialize_inputs();
     
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    gettimeofday(&start, &tzdummy);
    
    matrixNorm();  
    
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);
    
    printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");
    
    exit(0);
}