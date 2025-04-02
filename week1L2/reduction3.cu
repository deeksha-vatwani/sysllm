#include<cuda_runtime.h>
#include<stdio.h>


#define ELEMENT_PER_BLOCK 256

__global__ void reductionKernel(int *d_input, int *d_output, int N) {
    __shared__ int sdata[ELEMENT_PER_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < N) ? d_input[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(d_output, sdata[0]);
    }
}

int main() {
    int N = 1024*1024*1024;
    int *d_input, *d_output;
    int *h_input, *h_output;

    h_input = (int*)malloc(N * sizeof(int));
    h_output = (int*)malloc(sizeof(int));
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, sizeof(int));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i % 10; // Example data
    }
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    // Launch kernel
    int blockSize = ELEMENT_PER_BLOCK;
    int numBlocks = (N + blockSize - 1) / blockSize;
    reductionKernel<<<numBlocks, blockSize>>>(d_input, d_output, N);
    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    // Print result
    printf("Sum: %d\n", *h_output);
    // Check result
    int expected_sum = 0;
    for (int i = 0; i < N; i++) {
        expected_sum += h_input[i];
    }
    if (*h_output == expected_sum) {
        printf("Result is correct!\n");
    } else {
        printf("Result is incorrect! Expected %d, got %d\n", expected_sum, *h_output);
    }

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

