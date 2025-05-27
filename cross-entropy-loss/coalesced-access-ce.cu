#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <cuda.h>

#define C 512
#define N 1048576
#define BLOCK_SIZE 32
#define WARP_SIZE 32

float ce_loss_sequential(float * L, int* Y){
    float total_log_likelyhood = 0;
    for (int i=0; i<N; ++i){
        float exp_logit = expf(L[i*C + Y[i]]);

        float softmax_denominator = 0;
        for (int z = 0; z<C; ++z){
            softmax_denominator += expf(L[i*C + z]);
        }
        float log_likelyhood = logf(exp_logit/softmax_denominator);
        total_log_likelyhood += log_likelyhood;
    }

    return -total_log_likelyhood / N;
}

float randomFloat(float min, float max) {
  return min + (float)rand() / RAND_MAX * (max - min);
}

void fill_L(float * L){
    for (int i=0; i<N; ++i){
        for (int j=0; j<C; ++j){
            L[i*C + j] = randomFloat(-1, 1);
        }
    }
}

void fill_Y(int* Y){
    for (int i=0; i<N; ++i){
        Y[i] = 0;
    }
}

__global__
void coalescedAccessCE(float * L, int* Y, float * loss){
    //One warp/block per row
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < C*N){
        float thread_sum = 0;

        __shared__ float L_row[C];
        __shared__ float row_exp_sum;

        if (threadIdx.x == 0){
            row_exp_sum = 0;
        }

        for (int p = 0; threadIdx.x + p*WARP_SIZE<C; ++p){
            L_row[threadIdx.x + p*WARP_SIZE] = L[i+p*WARP_SIZE];
            thread_sum += expf(L_row[threadIdx.x + p*WARP_SIZE]);
        }

        __syncthreads();

        //Make a warp reduce later
        for (int offset = 1; offset <= WARP_SIZE/2; offset *= 2) {
            thread_sum = thread_sum +  __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (threadIdx.x == 0) row_exp_sum = thread_sum;

        if(threadIdx.x == 0){
            int class_ind = Y[blockIdx.x];
            float row_loss = (-L_row[class_ind] + log(row_exp_sum))/N;
            
            float cur_loss = atomicAdd(loss, row_loss);
        }
    }
    
}

int main() {
    float * h_L = (float*)malloc(C*N*sizeof(float));fill_L(h_L);
    float * d_L;cudaMalloc(&d_L, C*N*sizeof(float));
    cudaMemcpy(d_L, h_L, C*N*sizeof(float), cudaMemcpyHostToDevice);

    int * h_Y = (int*)malloc(N*sizeof(int));fill_Y(h_Y);
    int * d_Y;cudaMalloc(&d_Y, N*sizeof(int));
    cudaMemcpy(d_Y, d_Y, N*sizeof(int), cudaMemcpyHostToDevice);

    float h_loss = 0; float* h_loss_p = &h_loss; 
    float* d_loss; 
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemcpy(d_loss, h_loss_p, sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // Kernel Timing
    cudaEventRecord(start);
    coalescedAccessCE<<<N, WARP_SIZE>>>(d_L, d_Y, d_loss);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel Execution Time ms: %f \n", ms);
    // END


    cudaMemcpy(h_loss_p, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // float correct_loss = ce_loss_sequential(h_L, h_Y);
    // printf("Correct Loss: %f, Kernel Loss: %f\n", correct_loss, *h_loss_p);

    cudaFree(d_loss);
    cudaFree(d_L);
    cudaFree(d_Y);
    free(h_L);
    free(h_Y);
}