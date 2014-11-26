/*
 * svd_main.c
 *
 *  Created on: Nov 17, 2014
 *      Author: c-hchang
 */
/*
 * CULA Example: GESVD
 *
 * This example is a svd image decomposition via the reduced-rank
 * reconstruction of an image
 * reference from CULA example image compression 
 * reconstruction image use CUBLAS and CULA
 *
 * This is accomplished via the following steps.
 *
 * 1. Read image via OpenCV interface
 * 2. Compute SVD of image.
 * 3. Reconstruct image from the previously computed
 *    SVD.
 */

#include <stdlib.h>
#include <cula_lapack_device.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include <cula_lapack.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))
#define MAX_ITER 200
#define IDX2C (i ,j , ld ) ((( j )*( ld ))+( i ))

void checkStatus(culaStatus status);
void generate_image(float* matrix, int width, int height, int nChannels  ,unsigned char* InputImage, int RGB);
void recompute_original_image(float* A_prime, float* S, float* U, float* VT, int M, int N);
void write_image(float* Inputimage, int M, int N, IplImage* output, int RGB);
double getHighResolutionTime(void);

int main(int argc, char** argv)
{
	double start_time, end_time;
	
  // read image via OpenCV
	IplImage* imginput = cvLoadImage("/c2_4.jpg",3);
	int  M= imginput->width;      // image width
  int  N= imginput->height;   // image height
	int WidthStep =   N;          // image height;
	int nChannels =   imginput->nChannels;
	int Depth     =   8;
	IplImage *output = cvCreateImage(cvSize(M, N), Depth, nChannels);

	unsigned char* pImage = (unsigned char* ) imginput->imageData;

	/* Declare all the necessary variables */
  /* Dimensions of matrices */

    printf("image size %d %d \n", M, N);

    culaStatus status;

    /* Setup SVD Parameters */
    int LDA  = M;
    int LDU  = M;
    int LDVT = N;
    /* create memory to store data (CPU side)*/
    float* A_R =  NULL;   float* A_G =  NULL;  float* A_B =  NULL;
    float* S_R =  NULL;   float* S_G =  NULL;  float* S_B =  NULL;
    float* U_R =  NULL;   float* U_G =  NULL;  float* U_B =  NULL;
    float* VT_R = NULL;   float* VT_G = NULL;  float* VT_B =  NULL;

    /* Memory space for reconstructed image */
    float* A_prime_R = NULL;
    float* A_prime_G = NULL;
    float* A_prime_B = NULL;
  

    float cula_time;
  
    /* create SVD parameters*/

    char jobu =  'A';
    char jobvt = 'A';

    /* create memory to store data RGB separately (CPU side)*/
    A_R = (float*)malloc(M*N*sizeof(float));
    S_R = (float*)malloc(imin(M,N)*sizeof(float));
    U_R = (float*)malloc(LDU*M*sizeof(float));
    VT_R = (float*)malloc(LDVT*N*sizeof(float));

    A_G = (float*)malloc(M*N*sizeof(float));
    S_G = (float*)malloc(imin(M,N)*sizeof(float));
    U_G = (float*)malloc(LDU*M*sizeof(float));
    VT_G = (float*)malloc(LDVT*N*sizeof(float));

    A_B = (float*)malloc(M*N*sizeof(float));
    S_B = (float*)malloc(imin(M,N)*sizeof(float));
    U_B = (float*)malloc(LDU*M*sizeof(float));
    VT_B = (float*)malloc(LDVT*N*sizeof(float));

    A_prime_R = (float*)malloc(M*N*sizeof(float));
    A_prime_G = (float*)malloc(M*N*sizeof(float));
    A_prime_B = (float*)malloc(M*N*sizeof(float));

    
    if(!A_R || !S_R || !U_R || !VT_R) //Memory allocation failed
    {
        free(A_R);
        free(U_R);
        free(S_R);
        free(VT_R);

        return EXIT_FAILURE;
    }
    
  printf("Separate image to RGB data ... ");

	generate_image(A_R, M, N, nChannels, pImage, 0);
	generate_image(A_G, M, N, nChannels, pImage, 1);
	generate_image(A_B, M, N, nChannels, pImage, 2);

    //time(&begin_time);

    /* Initialize CULA */
    status = culaInitialize();
    checkStatus(status);

    /* Perform singular value decomposition CULA */
    printf("Performing singular value decomposition using CULA ... ");

    cudaEvent_t start,stop;

    //float elapsedTime;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start,0);

    start_time = getHighResolutionTime();
    
    /* cula svd*/
    printf("strat running svd on cula");
    status = culaSgesvd(jobu, jobvt, M, N, A_R, LDA, S_R, U_R, LDU, VT_R, LDVT);
    status = culaSgesvd(jobu, jobvt, M, N, A_G, LDA, S_G, U_G, LDU, VT_G, LDVT);
    status = culaSgesvd(jobu, jobvt, M, N, A_B, LDA, S_B, U_B, LDU, VT_B, LDVT);

    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    
    culaShutdown();

    //cudaEventRecord(stop,0);
    //cudaEventSynchronize(stop);

    //CALCULATE ELAPSED TIME
    //cudaEventElapsedTime(&elapsedTime,start,stop);

    //DISPLAY COMPUTATION TIME


    /*
     * Zero-out N trailing singular values and corresponding vectors.
     * For singular vector <, write 1 to Mth element of the vector.
    */
    
    recompute_original_image(A_prime_R, S_R, U_R, VT_R, M, N);
    recompute_original_image(A_prime_B, S_B, U_B, VT_B, M, N);
    recompute_original_image(A_prime_G, S_G, U_G, VT_G, M, N);

    /* Reconstruct Reduced image by A = U*S*VT */
     printf("Reconstructing reduced image ... ");
    
    /* write image to jpg image format */
    write_image(A_prime_R, M, N, output,0);
    write_image(A_prime_G, M, N, output,1);
    write_image(A_prime_B, M, N, output,2);

    /* Clean up workspace, input, and output memory allocations */
    free(A_R);   free(A_G);   free(A_B);
    free(U_R);   free(U_G);   free(U_B);
    free(S_R);   free(S_G);   free(S_B);
    free(VT_R);  free(VT_G);  free(VT_B);

    free(A_prime_R); free(A_prime_G);  free(A_prime_B);

    cvNamedWindow( "DWT Image", 1 ); // creation of a visualisation window
    cvShowImage( "DWT Image", output); // image visualisation

    cvSaveImage("/Users/c-hchang/Desktop/reconstructed_image.jpg",output, 0);
    cvWaitKey('x');

    return EXIT_SUCCESS;
}

/* Check for errors and exit if one occurred */
void checkStatus(culaStatus status)
{
    char buf[256];

    if(!status)
        return;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);

    culaShutdown();
    exit(EXIT_FAILURE);
}

/* Separate image data to RGB and convert it from Uchar to floating point*/
void generate_image(float* matrix, int width, int height, int nChannels  ,unsigned char* InputImage, int RGB)
{
    int WidthStep= height*nChannels ;   // for gray level scale
              // for gray level setting

    for (int i=0; i < width; i++)
     {
        for (int j =0; j < height; j++)
        {

            matrix[i*height + j] = (float) InputImage[i*WidthStep + j*nChannels + RGB];
        }
    }
    return ;
}
/* Use  cublasSgemm to speed up image reconstruction from SVD*/
void recompute_original_image(float* A_prime, float* S, float* U, float* VT, int M, int N)
{
    float zero = 0.0f;
    float one  = 1.0f;

    int i;
    float* U_v;
    float* A_prime_v;
    float* VT_v;

    float* V_matrix;
    float* S_matrixT;
    float* S_matrix;
    float* Re_matrix;

    cudaError_t cudaStat ;  // cudaMalloc status
    cublasStatus_t stat ;   // CUBLAS functions status
    cublasHandle_t handle ; // CUBLAS context

    float*Smatrix = NULL;
    Smatrix= malloc(M*N*sizeof(float));
    memset(Smatrix,0,M*N*sizeof(float));
    
     /* Put Singular value to M*N matrix, in order to run everything on GPU*/
      for (int i=0;i<M;i++)
       {
         if (i<N)  // only N singular values //
           Smatrix[(i*N)+i]= S[i];
         }

   /* GPU memory for SVD image reconstruction*/
   cudaMalloc((void**)&V_matrix , N * N * sizeof(float));
   cudaMalloc((void**)&S_matrix , M * N * sizeof(float));
   cudaMalloc((void **)&S_matrixT , M * N * sizeof(float));
   cudaMalloc((void **)&Re_matrix , M * N * sizeof(float));

   cudaMemcpy(V_matrix , VT , N * N * sizeof(float) , cudaMemcpyHostToDevice);
   cudaMemcpy(S_matrix , Smatrix , M * N * sizeof(float) , cudaMemcpyHostToDevice);
   cublasCreate(&handle);

   /* transpose S matrix for BLAS raw storage format  Sgeam */
   cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &one, S_matrix, N, &zero, S_matrix, M, S_matrixT, M);
   
   /*   S*VT matrix for multiplication of S*VT Sgemm   Result -> transposed */
   stat =  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, M, N, N, &one , S_matrixT, M, V_matrix,N, &zero, Re_matrix,M);

   cublasDestroy ( handle );
   stat =  cublasCreate (& handle ); // initialize CUBLAS context

   cudaStat = cudaMalloc (( void **)& U_v ,       M*M*sizeof(float)); // device
   cudaStat = cudaMalloc (( void **)& A_prime_v , M*N*sizeof(float)); // device
   stat = cublasSetMatrix (M,M, sizeof (float) , U , M , U_v , M);

   stat =  cublasCreate (& handle );

   /*   U_T*Result matrix for multiplication of S*VT Sgemm   Result -> transposed */
   stat =  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, M, N, M, &one , U_v, M, Re_matrix, M, &zero, A_prime_v,M);
   stat =  cublasGetMatrix( M,N, sizeof(float), A_prime_v ,M, A_prime ,M);
   cublasDestroy ( handle );

   cudaFree(S_matrix);  cudaFree(U_v);       cudaFree(A_prime_v);
   cudaFree(V_matrix);  cudaFree(S_matrixT); cudaFree(Re_matrix);

   free(Smatrix);


    return;
}

void write_image(float* Inputimage, int M, int N, IplImage* output, int RGB)
{

	 /* normalized value to 0~255*/
   float maxval=-99999999;
   int  colnum = N;   // shift N number;
	 //int  WidthStep = output-> widthStep ;
	 int  nChannels = output-> nChannels ;
	 int  Widthstep = nChannels*N;
      for (int i=0; i < M; i++)
  	    {
  	        for (int j =0; j < N; j++)
  	        {
  	         if (maxval <= (float) Inputimage[i*colnum+j])
  	         {
  	        	  maxval = Inputimage[i*colnum+j] ;
  	        }
  	     }
  	    }
     

    unsigned char  *pGreen = ( unsigned char* )output->imageData; 
    /* normaized to unsigned char and store data to opencv image format*/
	  for (int i=0; i < M; i++)
	    {
	        for (int j =0; j < N; j++)
	        {
	           if(Inputimage[i*colnum+j]<0)
	                (pGreen[i*Widthstep+j*nChannels + RGB]=(unsigned char) (0));
	            else
	             {

	            	pGreen[i*Widthstep + j*nChannels + RGB] = (unsigned  char) ((Inputimage[i*colnum+j]/maxval)*255.0);

	            }
	        }
	    }
    return;
}

double getHighResolutionTime(void)
{
    struct timeval tod;
    gettimeofday(&tod, NULL);
    double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
    return time_seconds;
}




