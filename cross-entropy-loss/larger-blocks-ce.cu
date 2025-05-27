#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <cuda.h>

#define C 512
#define N 1048576
#define BLOCK_HEIGHT 32
#define BLOCK_WIDTH 32
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
void multipleRowPerBlockCE(float * L, int* Y, float * loss){
    //One warp per row
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ float L_tile[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ float correct_label_logits[BLOCK_HEIGHT];
    __shared__ float negative_log_likelyhoods[BLOCK_HEIGHT];
    __shared__ float s_Y[BLOCK_HEIGHT];
    float thread_sum = 0;
    if (threadIdx.y == 0){
        //Later try threadIdx.y ==0 to minimize divergence
        //Adds complexity especially with non-square blocks
        negative_log_likelyhoods[threadIdx.y] = 0;
        s_Y[threadIdx.x] = Y[blockIdx.y * blockDim.y + threadIdx.x];
        correct_label_logits[threadIdx.y] = 0;
    }
    for (int phase=0; (phase+1)*WARP_SIZE < C; ++phase){
        if (row*C + phase*WARP_SIZE + threadIdx.x < C*N){
            L_tile[threadIdx.y][threadIdx.x] = L[row*C + phase*WARP_SIZE + threadIdx.x];
            thread_sum += expf(L_tile[threadIdx.y][threadIdx.x]);
        }
        __syncthreads();
    }
        
    __syncthreads();

    //Warpwise reduction (adding up threads in warp/row)
    for (int offset = 1; offset <= WARP_SIZE/2; offset *= 2) {
        thread_sum = thread_sum +  __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (threadIdx.x == 0) negative_log_likelyhoods[threadIdx.y] = -correct_label_logits[threadIdx.y] + logf(thread_sum);

    __syncthreads();

    // BlockWise Reduction (averaging across rows in block)
    // Same here only works with square but we will asume it is square
    for(int offset = blockDim.y/2; offset > 0; offset/=2){
        if (threadIdx.y == 0 && threadIdx.x < offset){
            negative_log_likelyhoods[threadIdx.x] += negative_log_likelyhoods[threadIdx.x + offset];
        }
        __syncthreads();
    }



    if(threadIdx.x == 0 && threadIdx.y == 0){
        atomicAdd(loss, negative_log_likelyhoods[0]/N);
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

    dim3 blockDim(WARP_SIZE, BLOCK_HEIGHT, 1);
    dim3 gridDim(1, ceil(N/BLOCK_HEIGHT), 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;



    // Kernel Timing
    cudaEventRecord(start);
    multipleRowPerBlockCE<<<gridDim, blockDim>>>(d_L, d_Y, d_loss);
    //Error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Synchronize and check for any errors during kernel execution
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //
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