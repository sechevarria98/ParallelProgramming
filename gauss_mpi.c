#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

#define MAXN 3000

// Removed volatile so that no
// warnings came from compile.
float A[MAXN][MAXN], B[MAXN], X[MAXN];

// Added both the id and nprocs variable for MPI
int N, id, nprocs = 2;

void gauss();
void initialize_work();

unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

void parameters(int argc, char **argv) {
	int seed = 0;
	srand(time_seed());

	if (argc == 3) {
		seed = atoi(argv[2]);
		srand(seed);
    if (id == 0) {
		  printf("Random seed = %i\n", seed);
    }
	} 
	if (argc >= 2) {
		N = atoi(argv[1]);
		if (N < 1 || N > MAXN) {
      if (id == 0) {
			  printf("N = %i is out of range.\n", N);
      }
			exit(0);
		}
	}
	else {
		printf("Usage: %s <matrix_dimension> [random seed]\n",
				argv[0]);    
		exit(0);
	}

  if (id == 0) {
	  printf("\nMatrix dimension N = %i.\n", N);
  }
}

void initialize_inputs() {
  int row, col;

  if (id == 0)
    printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

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
  struct timeval etstart, etstop;
  struct timezone tzdummy;
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;

  double start, end;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

  // Has to be called by each processor
  // so that each one initializes the matrix size
  parameters(argc, argv);

  if(id == 0) {

    // Only one processor needs to do these
    initialize_inputs();
    print_inputs();
    printf("\nStarting clock.\n");
    gettimeofday(&etstart, &tzdummy);
    times(&cputstart);

    start = MPI_Wtime();
  }

  initialize_work();
  gauss();

  if (id == 0) {

    // Only one processor needs to do these
    end = MPI_Wtime();

    gettimeofday(&etstop, &tzdummy);
    times(&cputstop);
    printf("Stopped clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

    print_X();

    printf("\nMPI Wtime --> %f seconds.\n", end - start);

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
  }
  
  MPI_Finalize();
  exit(0);
}

// ./serial.out 5 341
// mpirun -np 3 ./gauss_mpi.out 5 341
// mpicc gauss_mpi.c -o gauss_mpi.out
// mpirun -np <num treads> ./gauss_mpi.out <matrix size> [random seed]

void initialize_work() {
  
  // Barrier since proc 0 would take some
  // time to get to here
  MPI_Barrier(MPI_COMM_WORLD);

  // Only process 0 has values in both matrix A and B.
  // Process 0 should send each other process the 
  // necessary data to complete their work
  
  if (id == 0 && nprocs >= 2) {
    for (int proc = 1; proc < nprocs; proc++) {
      for (int row = proc; row < N; row += nprocs) {
        MPI_Send(&A[row], N, MPI_FLOAT, proc, 0, MPI_COMM_WORLD);
        MPI_Send(&B[row], 1, MPI_FLOAT, proc, 1, MPI_COMM_WORLD);
        // printf("processor -> %d sent over A[%d] and B[%d] to processor -> %d\n", id, row, row, proc);
      }
    }
  } else if (id != 0) {
    // Start with second process 
    for (int proc = 1; proc < nprocs; proc++) {
      // i increments over the number of procs so that no two
      // processes have the same data in their the matricies
      for (int row = proc; row < N; row += nprocs) {
        if (proc == id) {
          // avoiding the bottleneck of having multiple processes
          // calling MPI_Recv and awaiting proc 0 to send them data
          MPI_Recv(&A[row], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&B[row], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // printf("processor -> %d recieved A[%d] and B[%d] from processor -> 0\n", id, row, row);
      }
    }
  }
}

void gauss() {
  int norm, row, col;
  float multiplier;

  // Barrier since proc 0 and others would take some
  // time to get to here
  MPI_Barrier(MPI_COMM_WORLD);

  for (norm = 0; norm < N - 1; norm++) {

    // Need to broadcast the data contained in these
    // rows that correspond to norm, since each processor
    // would not have this data to complete the operations below. 
    MPI_Bcast(&A[norm], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B[norm], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (row = norm + 1; row < N; row++) {

      // if (id != 0 && A[row][0] != 0)
      //   printf("row -> %d | id -> %d | A[%d] -> %f\n", row, id, row, A[row][0]);

      if (id == 0) {
        // Each id corresponds to the row modulo number of processors.
        // While proc 0 has all the matrix data it should still only
        // work on the left over rows.
        if (row % nprocs == id) {
          multiplier = A[row][norm] / A[norm][norm];

          for (col = norm; col < N; col++) {
            A[row][col] -= A[norm][col] * multiplier;
          } 
          B[row] -= B[norm] * multiplier;
        } 
        else if (row == norm + 1) {
          // recieve the current row data on each iteration of norm
          MPI_Recv(&A[row], N, MPI_FLOAT, row % nprocs, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&B[row], 1, MPI_FLOAT, row % nprocs, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      } else {
        // Each id corresponds to the row modulo number of processors.
        // Blocks out other processors from working on rows where there
        // is no data present.
        if (row % nprocs == id) {
          multiplier = A[row][norm] / A[norm][norm];

          for (col = norm; col < N; col++) {
            A[row][col] -= A[norm][col] * multiplier;
          }
          B[row] -= B[norm] * multiplier;

          if (row == norm + 1) {
            // send the current row data on each iteration of norm
            MPI_Send(&A[row], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&B[row], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
          }
        }
      }
    } 
  }

  // Only proc 0 should have to do
  // the back substitution
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
