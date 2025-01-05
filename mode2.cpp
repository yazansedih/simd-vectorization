// mode2

#include <immintrin.h>  // portable to all x86 compilers
#include <stdio.h>
#include <time.h>

#define DATA float
#define SIZE 128
// size  128/256 / 512 

DATA __attribute__((aligned(16))) A[SIZE][SIZE] ;
DATA __attribute__((aligned(16))) B[SIZE] ;
DATA __attribute__((aligned(16))) result[SIZE] ;

double seconds() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec / 1000000000.0;
}

void initialize_array(DATA a[], int size) {
	for (int i = 0 ; i < size; i++) 
        a[i] = rand()%2;
}

void initialize_Matrix(DATA a[SIZE][SIZE], int size) {
	for (int i = 0 ; i < size; i++) 
        for(int j=0; j< size; j++) 
            a[i][j] = rand()%2;     
}


static void matvec_simple ( size_t n , float vec_c[ SIZE ] ,
const float mat_a[ SIZE ] [ SIZE ] , const float vec_b[ SIZE ] ) {
	for ( int i = 0 ; i < n ; i ++)
		for ( int j = 0 ; j < n ; j ++)
		    vec_c[ i ] += mat_a[ i ] [ j ] * vec_b[ j ];
}   

static void matvec_unrolled(size_t n , float vec_c [ SIZE ] ,
const float mat_a[ SIZE ] [ SIZE ] , const float vec_b [ SIZE ] ) {
	for ( int i = 0 ; i < n ; i ++)
		for ( int j = 0 ; j < n ; j += 4 )
			vec_c [ i ] += mat_a [ i ] [ j + 0 ] * vec_b [ j + 0 ]
			+ mat_a [ i ] [ j + 1 ] * vec_b [ j + 1 ]
			+ mat_a [ i ] [ j + 2 ] * vec_b [ j + 2 ]
			+ mat_a [ i ] [ j + 3] * vec_b [ j + 3] ;
}


void matvec_sse(DATA a[ SIZE][ SIZE ] ,  DATA b[ SIZE ]) {

    int i, j, k;
    __m128 X, Y, Z;// 32 bit *4 in parallel 
   
    for(i=0; i<SIZE; i++) {
        DATA prod = 0;
        Z[0] = Z[1] = Z[2] = Z[3] = 0;
    
        for(j=0; j<SIZE; j+=4) 
        {
            X = _mm_load_ps(&a[i][j]);
            Y = _mm_load_ps(&b[j]);
            X = _mm_mul_ps(X, Y);
            Z = _mm_add_ps(X, Z);
        }
        for(k=0; k<4; k++) 
        {
            prod += Z[k];
        }
        result[i] = prod;
    }
}  

void print_result(){
     for (int i=0;i<SIZE;i++) printf("%f\t",result[i]);
}  

int main() {

	double before,after;
    DATA* r;
    
	initialize_Matrix(A,SIZE);
	initialize_array(B,SIZE);

	before = seconds();
	matvec_unrolled(SIZE,result,A,B);
	after = seconds();
	printf("Time:%f\n",after-before);

	before = seconds();
    matvec_sse(A,B);
	after = seconds();
	printf("Time:%f\n",after-before);

    return 0;
}