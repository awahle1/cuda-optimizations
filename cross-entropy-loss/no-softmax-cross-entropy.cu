#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

#define C 32
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
void noSoftmaxCrossEntropy(float * L, int* Y, float * loss){
    int i = threadIdx.x;
    
    //Log of Softmax
    float sum_exp = 0;
    for (int j=0; j<C; ++j){
        sum_exp += expf(L[i*C + j]);
    }
    int class_ind = Y[i];
    float negative_log_likelyhood = -L[i*C + class_ind] + log(sum_exp);

    atomicAdd(loss, negative_log_likelyhood/N);

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

    int blockSize = 32;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // Kernel Timing
    cudaEventRecord(start);
    noSoftmaxCrossEntropy<<<ceil(N/blockSize), blockSize>>>(d_L, d_Y, d_loss);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel Execution Time ms: %f \n", ms);
    // END


    cudaMemcpy(h_loss_p, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    float correct_loss = ce_loss_sequential(h_L, h_Y);
    printf("Correct Loss: %f, Kernel Loss: %f\n", correct_loss, *h_loss_p);

    cudaFree(d_loss);
    cudaFree(d_L);
    cudaFree(d_Y);
    free(h_L);
    free(h_Y);
}