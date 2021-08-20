#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int generate_random_dense_matrix(int M, int N, float **outA, float density)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            int r = rand();
            double dr = (double)r;
            float *curr = A + (j * M + i);

            if (dr / rMax > density)
            {
                *curr = 0.0f;
            }
            else
            {
                *curr = (dr / rMax) * 100.0;
            }

            if (*curr != 0.0f)
            {
                totalNnz++;
            }
        }
    }

    *outA = A;
    return totalNnz;
}

int main(void) {
    // Q, V: [seq-len, head-size] = [512, 64]
    // K: [head-size, seq-len] = [64, 512]
    // S: [seq-len, seq-len] = [512, 512]

    int SEQ_LEN = 512;
    int HEAD_SIZE = 64;
    float ATTN_DENSITY = 0.1;
    int QKV_SIZE = SEQ_LEN * HEAD_SIZE;

    int ldk = SEQ_LEN;
    int ldq = HEAD_SIZE;
    int ldv = SEQ_LEN;
    int ldo = SEQ_LEN;

    float alpha = 1.0f;
    float beta = 0.0f;

    float *dQ, *dV, *dK, *dS, *dCsrValS, *dO;
    float *hQ, *hV, *hK, *hS;
    int *dCsrRowPtrS, *dCsrColIndS, *dSNnzPerRow;
    void *dBuffer1, *dBuffer2, *dBuffer3;
    size_t bufferSize = 0;

    cusparseSpMatDescr_t Sdescr;
    cusparseDnMatDescr_t Kdescr, Qdescr, Vdescr, Odescr, SdescrDense;
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    float time_kernel;
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync));

    size_t N_REPEAT = 10000;



    //// initialize dense Q, K, V, O
    hQ = (float *)calloc(QKV_SIZE, sizeof(float));
    hV = (float *)calloc(QKV_SIZE, sizeof(float));
    hK = (float *)calloc(QKV_SIZE, sizeof(float));

    CHECK_CUDA(cudaMalloc((void **)&dQ, QKV_SIZE * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void**) &dV, QKV_SIZE * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void**) &dK, QKV_SIZE * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void**) &dO, QKV_SIZE * sizeof(float)))

    CHECK_CUDA(cudaMemcpy(dQ, hQ, QKV_SIZE * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dV, hV, QKV_SIZE * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dK, hK, QKV_SIZE * sizeof(float), cudaMemcpyHostToDevice))

    CHECK_CUSPARSE(cusparseCreateDnMat(&Kdescr, SEQ_LEN, HEAD_SIZE, ldk, dK, CUDA_R_32F, CUSPARSE_ORDER_COL))
    CHECK_CUSPARSE(cusparseCreateDnMat(&Qdescr, HEAD_SIZE, SEQ_LEN, ldq, dQ, CUDA_R_32F, CUSPARSE_ORDER_COL))
    CHECK_CUSPARSE(cusparseCreateDnMat(&Vdescr, SEQ_LEN, HEAD_SIZE, ldv, dV, CUDA_R_32F, CUSPARSE_ORDER_COL))
    CHECK_CUSPARSE(cusparseCreateDnMat(&Odescr, SEQ_LEN, HEAD_SIZE, ldo, dO, CUDA_R_32F, CUSPARSE_ORDER_COL))



    //// initialize sparsity mask S in CSR format
    int totalSNnz = generate_random_dense_matrix(SEQ_LEN, SEQ_LEN, &hS, ATTN_DENSITY);

    CHECK_CUDA(cudaMalloc((void **)&dS, sizeof(float) * SEQ_LEN * SEQ_LEN));
    CHECK_CUDA(cudaMalloc((void **)&dSNnzPerRow, sizeof(int) * SEQ_LEN));
    CHECK_CUDA(cudaMalloc((void **)&dCsrValS, sizeof(float) * totalSNnz));
    CHECK_CUDA(cudaMalloc((void **)&dCsrRowPtrS, sizeof(int) * (SEQ_LEN + 1)));
    CHECK_CUDA(cudaMalloc((void **)&dCsrColIndS, sizeof(int) * totalSNnz));

    CHECK_CUDA(cudaMemcpy(dS, hS, sizeof(float) * SEQ_LEN * SEQ_LEN, cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseCreateDnMat(&SdescrDense, SEQ_LEN, SEQ_LEN, SEQ_LEN, dS, CUDA_R_32F, CUSPARSE_ORDER_ROW))
    CHECK_CUSPARSE(cusparseCreateCsr(&Sdescr, SEQ_LEN, SEQ_LEN, 0,
                                     dCsrRowPtrS, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, SdescrDense, Sdescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer3, bufferSize))
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, SdescrDense, Sdescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer3))
    CHECK_CUSPARSE(cusparseCsrSetPointers(Sdescr, dCsrRowPtrS, dCsrColIndS, dCsrValS))
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, SdescrDense, Sdescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer3))



    /// Allocate external buffers for SDDMM and SpMM
    CHECK_CUSPARSE(cusparseConstrainedGeMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, Kdescr, Qdescr, &beta, Sdescr, 
        CUDA_R_32F, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize))

    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, Sdescr, Vdescr, &beta, Odescr, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize))


    /// Run SDDMM and SpMM
    // warmup
    CHECK_CUSPARSE(cusparseConstrainedGeMM(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, Kdescr, Qdescr, &beta, Sdescr,
        CUDA_R_32F, dBuffer1))

    CHECK_CUSPARSE(cusparseSpMM(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, Sdescr, Vdescr, &beta, Odescr, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, dBuffer2))


    CHECK_CUDA(cudaEventRecord(start_event, 0));

    for (size_t i = 0; i < N_REPEAT; i++) {
        CHECK_CUSPARSE(cusparseConstrainedGeMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, Kdescr, Qdescr, &beta, Sdescr,
            CUDA_R_32F, dBuffer1))

        CHECK_CUSPARSE(cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, Sdescr, Vdescr, &beta, Odescr, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer2))
    }

    CHECK_CUDA(cudaEventRecord(stop_event, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_event));
    CHECK_CUDA(cudaEventElapsedTime(&time_kernel, start_event, stop_event));
    printf("kernel:\t\t%.4f ms\n", time_kernel / N_REPEAT);



    /// cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(Sdescr))
    CHECK_CUSPARSE(cusparseDestroyDnMat(Kdescr))
    CHECK_CUSPARSE(cusparseDestroyDnMat(Qdescr))
    CHECK_CUSPARSE(cusparseDestroyDnMat(Vdescr))
    CHECK_CUSPARSE(cusparseDestroyDnMat(Odescr))
    CHECK_CUSPARSE(cusparseDestroyDnMat(SdescrDense))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    CHECK_CUDA(cudaFree(dQ))
    CHECK_CUDA(cudaFree(dK))
    CHECK_CUDA(cudaFree(dV))
    CHECK_CUDA(cudaFree(dO))
    CHECK_CUDA(cudaFree(dS))
    CHECK_CUDA(cudaFree(dCsrColIndS))
    CHECK_CUDA(cudaFree(dCsrRowPtrS))
    CHECK_CUDA(cudaFree(dCsrValS))
    CHECK_CUDA(cudaFree(dSNnzPerRow))
    CHECK_CUDA(cudaFree(dBuffer1))
    CHECK_CUDA(cudaFree(dBuffer2))



    return EXIT_SUCCESS;
}

