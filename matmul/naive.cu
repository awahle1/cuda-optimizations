#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

#define M 16
#define K 8
#define N 32


__global__ 
void matMul(float* A, float*B, float*C){
    int out_col = threadIdx.x + blockIdx.x * blockDim.x;
    int out_row = threadIdx.y + blockIdx.y * blockDim.y;

    if (out_row >= M || out_col >= N){
        return;
    }

    float accum = 0;

    for(int k = 0; k<K; ++k){
        accum += A[k + out_row * K]*B[out_col + k*N];
    }

    C[out_col + out_row*N] = accum;
}


void sequentialMatMul(float* A, float* B, float* C){
    for(int m=0; m<M; ++m){
        for(int n=0; n<N; ++n){
            float accum = 0;
            for(int k=0; k<K; ++k){
                accum += A[k + m*K]*B[k*N + n];
            }
            C[n + m*N] = accum;
        }
    }
}

float randomFloat(float min, float max) {
  return min + (float)rand() / RAND_MAX * (max - min);
}

void randomizeMatrix(float* V, int h, int w){
    for(int y=0; y<h; ++y){
        for(int x=0; x<w; ++x){
            V[x+ y*w] = randomFloat(-1,1);
        }
    }
}

void showMat(float* V, int h, int w){
    for(int y=0; y<h; ++y){
        for(int x=0; x<w; ++x){
            printf("%f ", V[x + y*w]);
        }
        printf("\n");
    }
}

int compMat(float* A,float* B, int h, int w, float error){
    for(int y=0; y<h; ++y){
        for(int x=0; x<w; ++x){
            if (A[x + y*w] - B[x + y*w] > error){
                return 1;
            }
        }
    }
    return 0;
}

int main(){
    float* A_h = (float*)malloc(M*K*sizeof(float));randomizeMatrix(A_h,M,K);
    float* B_h = (float*)malloc(K*N*sizeof(float));randomizeMatrix(B_h,K,N);
    float* C_h = (float*)malloc(M*N*sizeof(float));randomizeMatrix(C_h,M,N);
    float* C_seq = (float*)malloc(M*N*sizeof(float));randomizeMatrix(C_h,M,N);

    sequentialMatMul(A_h, B_h, C_seq);

    float* A_d=nullptr; float* B_d=nullptr; float* C_d = nullptr;
    cudaMalloc(&A_d, M*K*sizeof(float));
    cudaMalloc(&B_d, K*N*sizeof(float));
    cudaMalloc(&C_d, M*N*sizeof(float));

    cudaMemcpy(A_d, A_h, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, M*N*sizeof(float), cudaMemcpyHostToDevice);

    int block_size_x = 32;
    int block_size_y = 32;
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(((N-1)/block_size_x)+1, ((M-1)/block_size_y)+1, 1);

    matMul<<<blockDim, gridDim>>>(A_d, B_d, C_d);

    cudaMemcpy(C_h, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d);cudaFree(B_d);cudaFree(C_d);
    
    printf("%i\n", compMat(C_h, C_seq, M, N, 0.0001));
    free(A_h);free(B_h);free(C_h);
    

    return 0;
}