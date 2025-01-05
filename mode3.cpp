// mode3

#include <immintrin.h>  // portable to all x86 compilers
#include <stdio.h>
#include <time.h>

#define DATA float
#define SIZE 256
// size  128 / 256 / 512 

DATA __attribute__((aligned(16))) A[SIZE][SIZE] ;
DATA __attribute__((aligned(16))) B_T[SIZE][SIZE] ;
DATA __attribute__((aligned(16))) B[SIZE][SIZE] ;
DATA __attribute__((aligned(16))) result[SIZE][SIZE] ;

static void transpose(DATA matrix[SIZE][SIZE]) {
    for(int i=0; i<SIZE; i++)
        for(int j=0; j<SIZE; j++)
            B_T[j][i] = matrix[i][j];
}

double seconds() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec / 1000000000.0;
}

void initialize_Matrix(DATA a[SIZE][SIZE], int size) {
	for (int i = 0 ;  i < size ; i++)
        for(int j=0; j< size; j++) 
            a[i][j] = rand()%2;      
}

static void matmat ( size_t n , float mat_c [ SIZE ] [ SIZE ] ,const float mat_a [ SIZE ] [ SIZE ] , const float mat_b [ SIZE ] [SIZE ] ) {
	for ( int i = 0 ; i < n ; i ++) 
        for ( int k = 0 ; k < n ; k ++) 
            for ( int j = 0 ; j < n ; j ++) 
                mat_c [ i ] [ j ] += mat_a [ i ] [ k ] * mat_b [ k ] [ j ] ;
}    

void matmat_sse( DATA a [ SIZE ] [ SIZE ] ,  DATA b [ SIZE] [ SIZE] ) {
    int i, j, k,q;
    __m128 X, Y, Z;
   
    for(i=0; i<SIZE; i+=1) {
        for(j=0; j<SIZE; j+=1) {
            DATA prod = 0;
            Z[0] = Z[1] = Z[2] = Z[3] = 0;
        
            for(q=0; q<SIZE; q+=4) 
            {
                X = _mm_load_ps(&a[i][q]);
                Y = _mm_load_ps(&b[j][q]);
                X = _mm_mul_ps(X, Y);
                Z = _mm_add_ps(X, Z);
            }
            for(k=0; k<4; k++) 
                 prod += Z[k];
            
            result[i][j]=prod;
        }
    }
} 
  
int main()
{
	double before,after;

	initialize_Matrix(A,SIZE);
	initialize_Matrix(B,SIZE);
    transpose(B);

	before = seconds();
	matmat(SIZE,result,A,B);
	after = seconds();
	printf("Time:%f\n",after-before);

	before = seconds();
    matmat_sse(A,B_T);
	after = seconds();
	printf("Time:%f\n",after-before);

    return 0;
}
