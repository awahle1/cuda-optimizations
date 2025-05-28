#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <float.h>

#define C 512
#define N 1048576

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
void naiveCrossEntropy(float * L, int* Y, float * loss){
    //One Block Per Row
    int row = blockIdx.x;

    if (row>=N){
        return;
    }

    __shared__ int class_ind;
    __shared__ float logit_val;
    __shared__ float sum_exp; 
    if (threadIdx.x == 0) {
        class_ind = Y[row];
        sum_exp = 0.0f;
    }

    // __syncthreads();

    float thread_sum = 0; 
    for (int phase=0; phase<C/blockDim.x; ++phase){
        if (threadIdx.x + phase*blockDim.x < C){
            float val = expf(L[row*C + threadIdx.x + phase*blockDim.x]);
            thread_sum += val;
            if (threadIdx.x + phase*blockDim.x == class_ind){
                logit_val = val;
            }
        }
        
    }

    // atomicAdd(&sum_exp, thread_sum);

    // __syncthreads();

    // if (threadIdx.x == 0){
    //     float log_likelihood = logf(logit_val/sum_exp);
    //     atomicAdd(loss, -log_likelihood/N);
    // }


}

int main() {
    float * h_L = (float*)malloc(C*N*sizeof(float));fill_L(h_L);
    float * d_L;cudaMalloc(&d_L, C*N*sizeof(float));
    cudaMemcpy(d_L, h_L, C*N*sizeof(float), cudaMemcpyHostToDevice);

    int * h_Y = (int*)malloc(N*sizeof(int));fill_Y(h_Y);
    int * d_Y;cudaMalloc(&d_Y, N*sizeof(int));
    cudaMemcpy(d_Y, d_Y, N*sizeof(int), cudaMemcpyHostToDevice);

    float* d_loss; 
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    int blockSize = 32;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // Kernel Timing
    cudaEventRecord(start);
    naiveCrossEntropy<<<N, blockSize>>>(d_L, d_Y, d_loss);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel Execution Time ms: %f \n", ms);
    // END

    float zero = 0;
    float* h_loss = &zero; 
    cudaMemcpy(h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    float correct_loss = ce_loss_sequential(h_L, h_Y);
    printf("Correct Loss: %f, Kernel Loss: %f\n", correct_loss, *h_loss);

    cudaFree(d_loss);
    cudaFree(d_L);
    cudaFree(d_Y);
    free(h_L);
    free(h_Y);
}