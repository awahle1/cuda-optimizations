#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

#define C 8
#define N 2

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
            L[i*C + j] = randomFloat(-5, 5);
        }
    }
}

void fill_Y(int* Y){
    for (int i=0; i<N; ++i){
        Y[i] = 0;
    }
}

int main() {
    // float * h_L = (float*)malloc(C*N*sizeof(float));
    // fill_h(h_L);
    float h_L[] = {-0.8306562304496765, 1.6165248155593872, -1.1896085739135742, -1.680414080619812, 1.8756650686264038, -0.3747663199901581, 0.8855074048042297, 1.4021867513656616, -0.5551024675369263, 0.040698129683732986, -1.5343101024627686, -0.8756691813468933, -0.44812169671058655, -0.30535200238227844, 0.30701500177383423, -1.8096588850021362};


    // int * h_Y = (int*)malloc(N*sizeof(int));
    // fill_Y(h_Y);
    int h_Y[] = {5, 3};

    float ce_loss = ce_loss_sequential(h_L, h_Y);

    printf("Loss: %f", ce_loss);

}