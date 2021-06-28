#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N;  /* Matrix size */
int numThreads; /* Number of threads */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];

struct thread_args {
    int id;
    int norm;
} thread_args;

void gauss();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
	int seed = 0; 
	srand(time_seed());

	if (argc == 4) {
		seed = atoi(argv[2]);
		srand(seed);
		numThreads=atoi(argv[3]);
		printf("Random seed = %i\n", seed);
	} 
	if (argc >= 2) {
		N = atoi(argv[1]);
		if (N < 1 || N > MAXN) {
			printf("N = %i is out of range.\n", N);
			exit(0);
		}
	}
	else {
		printf("Usage: %s <matrix_dimension> [random seed]\n",
				argv[0]);    
		exit(0);
	}

	/* Print parameters */
	printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;
  struct timezone tzdummy;
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;

  parameters(argc, argv);
  initialize_inputs();
  print_inputs();

  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  times(&cputstart);

  pthread_t threads[numThreads];
  struct thread_args* args = malloc(numThreads * sizeOf(struct thread_args));

  int norm, index;
  for (int index = 0; index < numThreads; index++) {
    for (norm = index; norm < N - 1; norm+=numThreads) {
        args[index].id = index;
        args[index].norm = norm;
        pthread_create(&threads[index], NULL, gauss, (void *) &args[index]); 
    }
  }

  for (index = 0; index < numThreads; index++) {
    pthread_join(threads[index], NULL);
  }

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  print_X();

  printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);

  printf("(CPU times are accurate to the nearest %g ms)\n",
	 1.0/(float)CLOCKS_PER_SEC * 1000.0);
  printf("My total CPU time for parent = %g ms.\n",
	 (float)( (cputstop.tms_utime + cputstop.tms_stime) -
		  (cputstart.tms_utime + cputstart.tms_stime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My system CPU time for parent = %g ms.\n",
	 (float)(cputstop.tms_stime - cputstart.tms_stime) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My total CPU time for child processes = %g ms.\n",
	 (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
		  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("--------------------------------------------\n");
  
  exit(0);
}

/* ------------------ Above Was Provided --------------------- */
void *gauss(void * args) {
  int norm, row, col;
  float multiplier;

  struct thread_args* thread_a = (struct thread_args*) args;
  int id = thread_a->id;
  norm = thread_a->norm;

  /* Gaussian elimination */
    for (row = norm + 1; row < N; row++) {
        multiplier = A[row][norm] / A[norm][norm];
        for (col = norm; col < N; col++) {
            A[row][col] -= A[norm][col] * multiplier;
        }
        B[row] -= B[norm] * multiplier;
    }

  if (id == 0) {
    for (row = N - 1; row >= 0; row--) {
        X[row] = B[row];
        for (col = N-1; col > row; col--) {
        X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }
  }
}