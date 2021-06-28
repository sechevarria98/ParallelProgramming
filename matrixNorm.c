// nvcc matrixNorm_CUDA.cu -lm -o norm.out
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

/* Program Parameters */
#define N 6000 /* Matrix size */

// pointers to the device arrays
float *a_d, *b_d;

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

// printing parts of the larger matrices for testing 
void print_A() {
    printf("\nA = ");
    int printing = 10;
    if (N < 99) {
        printing = N;
    }
    for (int row = 0; row < printing; row++) {
        printf("\n[ ");
        for (int col = 0; col < printing; col++) {
            printf(" %5.2f ", A[row][col]);
        }
        printf("]\n");
    }
}

void print_B() {
    printf("\nB = ");
    int printing = 10;
    if (N < 99) {
        printing = N;
    }
    for (int row = 0; row < printing; row++) {
        printf("\n[ ");
        for (int col = 0; col < printing; col++) {
            printf(" %5.2f ", B[row][col]);
        }
        printf("]\n");
    }
}

/* Kernel function */

// variables are used to indentify and differentiate between GPU threads
// blockDim -> dimensions of each thread block
// blockIdx -> index of thread block within grid
// threadIdx -> index of thread withing thread block 

__global__ void matrixNorm(float *temp_a, float *temp_b) {
    // i is the unique thread id
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int row;
    float mu, sigma; // Mean and Standard Deviation
    
    // if (i == 1) {
    //     for (int k = 0; k < 10; k++) {
    //         printf(" %5.2f ", temp_a[k]);
    //     }
    //     printf("\n");
    // }

    // since i is used in array access only allow those
    // less than N so no out-of-bounds access happens
    if (i < N) {
        // printf("Thread num -> %d\n", i);
        mu = 0.0;

        // temp_a is the one-dimensional form of A
        // row * N is used as the starting point for each of the rows
        // + i would access the individual elements in each section
        for (row=0; row < N; row++)
            mu += temp_a[row * N + i];

        mu /= (float) N;
        sigma = 0.0;
        
        for (row=0; row < N; row++)
            sigma += powf(temp_a[row * N + i] - mu, 2.0);

        sigma /= (float) N;
        sigma = sqrt(sigma);

        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                temp_b[row * N + i] = 0.0;
            else
                temp_b[row * N + i] = (temp_a[row * N + i] - mu) / sigma;
        }   
    }
    
    __syncthreads();
}



int main(int argc, char **argv) {
    /* Timing variables */
    struct timeval start, stop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long runtime;

    /* Initialize A and B */
    initialize_inputs();

    // size would be an N by N matrix of floats
    float size = N*N*sizeof(float);

    // print_A();

    /* Start Clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    gettimeofday(&start, &tzdummy);

    cudaEvent_t st, sp;
    cudaEventCreate(&st);
    cudaEventCreate(&sp);

    // Begin the timer
    cudaEventRecord(st);

    // allocate memory for the device arrays
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);

    // copy A to device array which would be a one dimensional array
    cudaMemcpy(a_d, (void **) A, size, cudaMemcpyHostToDevice);
    
    /* Matrix Normalization */
    // <<<number of thread blocks, number of threads per block>>>

    // (N + 255)/256 would get the number of thread blocks needed
    // to access the elements in the array and in case 
    // its not evenly divided -> floor()

    matrixNorm<<<floor((N+255)/256), 256>>>(a_d, b_d);

    // copy device b_d to B
    cudaMemcpy((void **) B, b_d, size, cudaMemcpyDeviceToHost);

    // End the timer
    cudaEventRecord(sp);

    // Free the device arrays
    cudaFree(a_d);
    cudaFree(b_d);
    /* Stop Clock */
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);
    // print_B();
    
    float milliseconds = 0;
    // return time elapsed between st and sp
    cudaEventElapsedTime(&milliseconds, st, sp);
    /* Display timing results */
    printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
    printf("CUDA Timer = %g ms.\n", milliseconds);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");
    
    exit(0);
}