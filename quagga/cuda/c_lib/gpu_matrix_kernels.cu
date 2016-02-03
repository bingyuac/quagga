/*-
 * Copyright 2015 Grammarly, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <cuda_runtime.h>


#define MAX_NUM_THREADS_PER_BLOCK 512
#define MAX_NUM_BLOCKS_PER_KERNEL 128
#define FLT_MAX 3.402823466E+38F


__global__  void sliceColumns(int nrows,
                              int ncols,
                              const int* __restrict__ embedding_column_indxs,
                              const float* __restrict__ embedding_matrix,
                              float* __restrict__ dense_matrix) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    int dense_column_idx;
    int row_idx;
    int embedding_offset;
    for (int i = start_i; i < nelems; i += nthreads) {
        dense_column_idx = i / nrows;
        row_idx = i % nrows;
        embedding_offset = embedding_column_indxs[dense_column_idx] * nrows + row_idx;
        dense_matrix[i] = embedding_matrix[embedding_offset];
    }
}


__global__  void sliceColumnsAndTranspose(int nrows,
                                          int ncols,
                                          const int* __restrict__ embedding_column_indxs,
                                          const float* __restrict__ embedding_matrix,
                                          float* __restrict__ dense_matrix) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;
    int dense_row_idx;
    int dense_col_idx;
    int dense_offset;
    int embedding_offset;
    for (int i = start_i; i < nelems; i += nthreads) {
        dense_row_idx = i / ncols;
        dense_col_idx = i % ncols;
        dense_offset = dense_col_idx * nrows + dense_row_idx;
        embedding_offset = embedding_column_indxs[dense_row_idx] * ncols + dense_col_idx;
        dense_matrix[dense_offset] = embedding_matrix[embedding_offset];
    }
}


template <typename T>
__global__  void sliceRows(int embedding_matrix_nrows,
                           const int* __restrict__ embedding_row_indxs,
                           const T* __restrict__ embedding_matrix,
                           int nrows,
                           int ncols,
                           T* __restrict__ dense_matrix) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    int dense_column_idx;
    int row_idx;
    int embedding_offset;
    for (int i = start_i; i < nelems; i += nthreads) {
        dense_column_idx = i / nrows;
        row_idx = i % nrows;
        embedding_offset = dense_column_idx * embedding_matrix_nrows + embedding_row_indxs[row_idx];
        dense_matrix[i] = embedding_matrix[embedding_offset];
    }
}


__global__  void sliceRowsBatch(const int* embd_rows_indxs,
                                int nrows,
                                int ncols,
                                const float* __restrict__ embd_matrix,
                                int embd_nrows,
                                int embd_ncols,
                                float* __restrict__ dense_matrices[]) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * embd_ncols;
    const int total_nelems = nelems * ncols;

    int k, dense_offset, embd_row_idx, embd_col_idx, embd_offset;
    for (int i = start_i; i < total_nelems; i += nthreads) {
        k = i / nelems;
        dense_offset = i % nelems;
        embd_row_idx = embd_rows_indxs[k * nrows + i % nrows];
        embd_col_idx = dense_offset / nrows;
        embd_offset = embd_col_idx * embd_nrows + embd_row_idx;
        dense_matrices[k][dense_offset] = embd_matrix[embd_offset];
    }
}


__global__  void slicedRowsBatchScaledAdd(const int* embd_rows_indxs,
                                          int nrows,
                                          int ncols,
                                          float alpha,
                                          const float* __restrict__ dense_matrices[],
                                          int embd_nrows,
                                          int embd_ncols,
                                          float* __restrict__ embd_matrix) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * embd_ncols;
    const int total_nelems = nelems * ncols;

    int k, dense_offset, embd_row_idx, embd_col_idx, embd_offset;
    for (int i = start_i; i < total_nelems; i += nthreads) {
        k = i / nelems;
        dense_offset = i % nelems;
        embd_row_idx = embd_rows_indxs[k * nrows + i % nrows];
        embd_col_idx = dense_offset / nrows;
        embd_offset = embd_col_idx * embd_nrows + embd_row_idx;
        atomicAdd(embd_matrix + embd_offset, alpha * dense_matrices[k][dense_offset]);
    }
}


__global__ void columnArgmax(int nrows,
                             int ncols,
                             const float* __restrict__ a,
                             int* __restrict__ indxs) {
    __shared__ float maxVals[32];
    __shared__ int maxIndxs[32];
    float max_val = -FLT_MAX;
    int max_idx = 0;
    float val = 0;
    for (int i = threadIdx.x; i < ncols; i += 32) {
        val = a[blockIdx.x + i * nrows];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    maxVals[threadIdx.x] = max_val;
    maxIndxs[threadIdx.x] = max_idx;
    __syncthreads();

    if (threadIdx.x == 0) {
        max_val = -FLT_MAX;
        max_idx = 0;
        for (int i = 0; i < 32; i++)
            if (maxVals[i] > max_val) {
                max_val = maxVals[i];
                max_idx = maxIndxs[i];
            }
        indxs[blockIdx.x] = max_idx;
    }
}


__global__ void hprodSum(int nelems,
                         int nrows,
                         const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        atomicAdd(C + i % nrows, A[i] * B[i]);
    }
}


__global__ void sumHprod(int nelems,
                         const float* __restrict__ A,
                         const float* __restrict__ B,
                         const float* __restrict__ C,
                         const float* __restrict__ D,
                         float* __restrict__ E) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        E[i] = A[i] * B[i] + C[i] * D[i];
    }
}


__global__ void sumHprod(int nelems,
                         const float* __restrict__ A,
                         const float* __restrict__ B,
                         const float* __restrict__ C,
                         const float* __restrict__ D,
                         const float* __restrict__ E,
                         float* __restrict__ F) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        F[i] = A[i] * B[i] * C[i] + D[i] * E[i];
    }
}


__global__ void sumHprod(int nelems,
                         const float* __restrict__ A,
                         const float* __restrict__ B,
                         const float* __restrict__ C,
                         const float* __restrict__ D,
                         const float* __restrict__ E,
                         const float* __restrict__ F,
                         const float* __restrict__ G,
                         const float* __restrict__ H,
                         const float* __restrict__ I,
                         const float* __restrict__ J,
                         const float* __restrict__ K,
                         float* __restrict__ L) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        L[i] = A[i] * B[i] * C[i] + D[i] * E[i] + F[i] * G[i] + H[i] * I[i] + J[i] * K[i];
    }
}


__global__  void hadamardProduct(int nelems,
                                 const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        c[i] = a[i] * b[i];
    }
}


__global__  void hadamardProduct(int nelems,
                                 const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 const float* __restrict__ c,
                                 float* __restrict__ d) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        d[i] = a[i] * b[i] * c[i];
    }
}


__global__  void addHadamardProduct(int nelems,
                                    const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float alpha,
                                    float* __restrict__ c) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        c[i] = a[i] * b[i] + alpha * c[i];
    }
}


__global__  void addHadamardProduct(int nelems,
                                    const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    const float* __restrict__ c,
                                    float alpha,
                                    float* __restrict__ d) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        d[i] = a[i] * b[i] * c[i] + alpha * d[i];
    }
}


__global__  void addScaledHadamardProduct(int nelems,
                                          const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float alpha,
                                          float beta,
                                          float* __restrict__ c) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        c[i] = alpha * c[i] + beta * a[i] * b[i];
    }
}



__global__  void addScaledColumnsSlice(int nrows,
                                       int ncols,
                                       float alpha,
                                       const float* __restrict__ dense_matrix,
                                       const int* __restrict__ embedding_column_indxs,
                                       float* __restrict__ embedding_matrix) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    int dense_column_idx;
    int row_idx;
    int embedding_offset;
    for (int i = start_i; i < nelems; i += nthreads) {
        dense_column_idx = i / nrows;
        row_idx = i % nrows;
        embedding_offset = embedding_column_indxs[dense_column_idx] * nrows + row_idx;
        atomicAdd(embedding_matrix + embedding_offset, alpha * dense_matrix[i]);
    }
}


__global__  void addScaledRowsSlice(int nrows,
                                    int ncols,
                                    float alpha,
                                    const float* __restrict__ dense_matrix,
                                    const int* __restrict__ embedding_row_indxs,
                                    int embd_nrows,
                                    float* __restrict__ embedding_matrix) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    int embd_col_idx;
    int embd_row_idx;
    int embd_offset;
    for (int i = start_i; i < nelems; i += nthreads) {
        embd_row_idx = embedding_row_indxs[i % nrows];
        embd_col_idx = i / nrows;
        embd_offset = embd_col_idx * embd_nrows + embd_row_idx;
        atomicAdd(embedding_matrix + embd_offset, alpha * dense_matrix[i]);
    }
}


__global__ void assignSum(int nelems,
                          const float* matrices[],
                          int n,
                          float* __restrict__ s) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        s[i] = 0.0f;
        for (int k = 0; k < n; k++) {
            s[i] += matrices[k][i];
        }
    }
}


__global__ void addSum(int nelems,
                       const float* matrices[],
                       int n,
                       float* __restrict__ s) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        for (int k = 0; k < n; k++) {
            s[i] += matrices[k][i];
        }
    }
}


__global__ void scale(int nelems,
                      const float* __restrict__ data,
                      float alpha,
                      float* __restrict__ out_data) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out_data[i] = alpha * data[i];
    }
}


__global__ void fill(int nelems,
                     float value,
                     float* __restrict__ out_data) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out_data[i] = value;
    }
}


__global__ void maskedFill(int nelems,
                           float value,
                           const float* __restrict__ mask,
                           float true_value,
                           float* __restrict__ out_data) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out_data[i] = (mask[i] == true_value) * value +
                      (mask[i] != true_value) * out_data[i];
    }
}


__global__ void matrixVectorRowAddition(int nrows,
                                        int ncols,
                                        const float* __restrict__ matrix,
                                        float alpha,
                                        const float* __restrict__ vector,
                                        float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] = matrix[i] + alpha * vector[i / nrows];
    }
}


__global__ void assignScaledAddition(int nelems,
                                     float alpha,
                                     const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] = alpha * (a[i] + b[i]);
    }
}


__global__ void assignScaledSubtraction(int nelems,
                                        float alpha,
                                        const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] = alpha * (a[i] - b[i]);
    }
}


__global__ void addScaledSubtraction(int nelems,
                                     float alpha,
                                     const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] += alpha * (a[i] - b[i]);
    }
}


__global__ void softmaxCeDerivative(int batchSize,
                                    int numClasses,
                                    const float* __restrict__ probs,
                                    const int* __restrict__ targetClasses,
                                    float* __restrict__ derivatives) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = batchSize * numClasses;

    for (int i = start_i; i < nelems; i += nthreads) {
        derivatives[i] = (probs[i] - (i / batchSize == targetClasses[i % batchSize])) / batchSize;
    }
}


__global__ void addSoftmaxCeDerivative(int batchSize,
                                       int numClasses,
                                       const float* __restrict__ probs,
                                       const int* __restrict__ targetClasses,
                                       float* __restrict__ derivatives) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = batchSize * numClasses;

    for (int i = start_i; i < nelems; i += nthreads) {
        derivatives[i] += (probs[i] - (i / batchSize == targetClasses[i % batchSize])) / batchSize;
    }
}



__global__ void assignSequentialMeanPooling(int nrows,
                                            int ncols,
                                            const float* matrices[],
                                            int n,
                                            float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;
    const int total_nelems = nelems * n;

    int k, m;
    for (int i = start_i; i < total_nelems; i += nthreads) {
        k = i / nelems;
        m = i % nelems;
        atomicAdd(out + m, matrices[k][m] / n);
    }
}


__global__ void assignSequentialSumPooling(int nrows,
                                           int ncols,
                                           const float* matrices[],
                                           int n,
                                           float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;
    const int total_nelems = nelems * n;

    int k, m;
    for (int i = start_i; i < total_nelems; i += nthreads) {
        k = i / nelems;
        m = i % nelems;
        atomicAdd(out + m, matrices[k][m]);
    }
}


__global__ void assignSequentialWeightedSum(int nrows,
                                            int ncols,
                                            const float* matrices[],
                                            const float* __restrict__ weights,
                                            int n,
                                            float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;
    const int total_nelems = nelems * n;

    int k, m;
    for (int i = start_i; i < total_nelems; i += nthreads) {
        k = i / nelems;
        m = i % nelems;
        atomicAdd(out + m, matrices[k][m] * weights[k * nrows + i % nrows]);
    }
}


__global__ void sequentiallyTile(int nelems,
                                 const float* __restrict__ a,
                                 float* matrices[],
                                 int n) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_nelems = nelems * n;

    int k, m;
    for (int i = start_i; i < total_nelems; i += nthreads) {
        k = i / nelems;
        m = i % nelems;
        matrices[k][m] = a[m];
    }
}


__global__ void dropout(int nelems,
                        float dropout_prob,
                        const float* __restrict__ data,
                        const float* __restrict__ uniform_data,
                        float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] = data[i] * (uniform_data[i] > dropout_prob);
    }
}


__global__ void maskZeros(int nelems,
                          const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] = a[i] * (b[i] != 0.0f);
    }
}


__global__ void addMaskZeros(int nelems,
                            const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] += a[i] * (b[i] != 0.0f);
    }
}


__global__ void assignMaskedAddition(int nelems,
                                     const float* __restrict__ mask,
                                     const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] = mask[i] * a[i] + (1.0f - mask[i]) * b[i];
    }
}


__global__ void assignMaskedAdditionColumnBroadcasted(int nrows,
                                                      int ncols,
                                                      const float* __restrict__ mask,
                                                      const float* __restrict__ a,
                                                      const float* __restrict__ b,
                                                      float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    float m;
    for (int i = start_i; i < nelems; i += nthreads) {
        m = mask[i % nrows];
        out[i] = m * a[i] + (1.0f - m) * b[i];
    }
}


__global__ void addHprodOneMinusMask(int nelems,
                                     const float* __restrict__ mask,
                                     const float* __restrict__ a,
                                     float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] += (1.0f - mask[i]) * a[i];
    }
}


__global__ void addHprodOneMinusMaskColumnBroadcasted(int nrows,
                                                      int ncols,
                                                      const float* __restrict__ mask,
                                                      const float* __restrict__ a,
                                                      float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] += (1.0f - mask[i % nrows]) * a[i];
    }
}


__global__ void matrixVectorColumnHprod(int nrows,
                                        int ncols,
                                        const float* __restrict__ matrix,
                                        const float* __restrict__ vector,
                                        float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] = matrix[i] * vector[i % nrows];
    }
}


__global__ void maskColumnNumbersRowWise(int nrows,
                                         int ncols,
                                         const int* __restrict__ numbers,
                                         float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] = (i / nrows < numbers[i % nrows]);
    }
}


__global__ void batchHorizontalStack(int n,
                                     int nrows,
                                     int xNcols,
                                     int yNcols,
                                     const float* xMatrices[],
                                     const float* yMatrices[],
                                     float* outMatrices[]) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int xNelems = nrows * xNcols;
    const int yNelems = nrows * yNcols;
    const int outNelems = xNelems + yNelems;
    const int totalNelems = n * outNelems;

    int k;
    int m;
    for (int i = start_i; i < totalNelems; i += nthreads) {
        k = i / outNelems;
        m = i % outNelems;
        if (m < xNelems) {
            outMatrices[k][m] = xMatrices[k][m];
        } else {
            outMatrices[k][m] = yMatrices[k][m - xNelems];
        }
    }
}


__global__ void batchHorizontalSplit(int n,
                                     int nrows,
                                     int xNcols,
                                     int yNcols,
                                     const float* matrices[],
                                     float* xMatrices[],
                                     float* yMatrices[]) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int xNelems = nrows * xNcols;
    const int yNelems = nrows * yNcols;
    const int outNelems = xNelems + yNelems;
    const int totalNelems = n * outNelems;

    int k;
    int m;
    for (int i = start_i; i < totalNelems; i += nthreads) {
        k = i / outNelems;
        m = i % outNelems;
        if (m < xNelems) {
            xMatrices[k][m] = matrices[k][m];
        } else {
            yMatrices[k][m - xNelems] = matrices[k][m];
        }
    }
}


__global__ void repeatAlongRow(int repeats,
                               int nrows,
                               int ncols,
                               const float* __restrict__ a,
                               float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    int row_idx;
    int col_idx;
    int offset;
    int col_stride = repeats * nrows;
    for (int j = 0; j < repeats; j++) {
        offset = nrows * j;
        for (int i = start_i; i < nelems; i += nthreads) {
            row_idx = i % nrows;
            col_idx = i / nrows;
            out[offset + col_stride * col_idx + row_idx] = a[i];
        }
    }
}


__global__ void addRepeatAlongRowDerivative(int repeats,
                                            const float* __restrict__ a,
                                            int nrows,
                                            int ncols,
                                            float* __restrict__ derivative) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    int row_idx;
    int col_idx;
    int offset;
    int col_stride = repeats * nrows;
    for (int j = 0; j < repeats; j++) {
        offset = nrows * j;
        for (int i = start_i; i < nelems; i += nthreads) {
            row_idx = i % nrows;
            col_idx = i / nrows;
            derivative[i] += a[offset + col_stride * col_idx + row_idx];
        }
    }
}


__global__ void repeatAlongCol(int repeats,
                               int nrows,
                               int ncols,
                               const float* __restrict__ a,
                               float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    int offset;
    for (int j = 0; j < repeats; j++) {
        offset = nelems * j;
        for (int i = start_i; i < nelems; i += nthreads) {
            out[offset + i] = a[i];
        }
    }
}


__global__ void addRepeatAlongColDerivative(int repeats,
                                            const float* __restrict__ a,
                                            int nrows,
                                            int ncols,
                                            float* __restrict__ derivative) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    int offset;
    for (int j = 0; j < repeats; j++) {
        offset = nelems * j;
        for (int i = start_i; i < nelems; i += nthreads) {
            derivative[i] += a[offset + i];
        }
    }
}

__global__  void addScaledDivSqrt(int nelems,
                                  float alpha,
                                  const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float epsilon,
                                  float* __restrict__ c) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        c[i] += alpha * a[i] / sqrtf(b[i] + epsilon);
    }
}


__global__ void clip(int nelems,
                     float min_value,
                     float max_value,
                     const float* __restrict__ data,
                     float* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nelems; i += nthreads) {
        out[i] = (min_value > data[i]) * min_value + (data[i] > max_value) * max_value + (min_value <= data[i] && data[i] <= max_value) * data[i];
    }
}

template <typename T>
__global__ void transpose(int nrows,
                          int ncols,
                          const T* __restrict__ in,
                          T* __restrict__ out) {
    const int nthreads = blockDim.x * gridDim.x;
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelems = nrows * ncols;

    int out_row_idx;
    int out_col_idx;
    for (int i = start_i; i < nelems; i += nthreads) {
        out_col_idx= i % nrows;
        out_row_idx = i / nrows;
        out[out_col_idx * ncols + out_row_idx] = in[i];
    }
}


extern "C" {
    cudaError_t _transposeFloat(cudaStream_t stream,
                                int nrows,
                                int ncols,
                                const float* __restrict__ in,
                                float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        transpose<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, in, out);
        return cudaGetLastError();
    }


    cudaError_t _transposeInt(cudaStream_t stream,
                              int nrows,
                              int ncols,
                              const int* __restrict__ in,
                              int* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        transpose<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, in, out);
        return cudaGetLastError();
    }


    cudaError_t _clip(cudaStream_t stream,
                      int nelems,
                      float min_value,
                      float max_value,
                      const float* __restrict__ data,
                      float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        clip<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, min_value, max_value, data, out);
        return cudaGetLastError();
    }

    cudaError_t _addScaledDivSqrt(cudaStream_t stream,
                                  int nelems,
                                  float alpha,
                                  const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float epsilon,
                                  float* __restrict__ c) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addScaledDivSqrt<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, alpha, a, b, epsilon, c);
        return cudaGetLastError();
    }


    cudaError_t _repeatAlongRow(cudaStream_t stream,
                                int repeats,
                                int nrows,
                                int ncols,
                                const float* __restrict__ a,
                                float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        repeatAlongRow<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(repeats, nrows, ncols, a, out);
        return cudaGetLastError();
    }

    cudaError_t _addRepeatAlongRowDerivative(cudaStream_t stream,
                                             int repeats,
                                             const float* __restrict__ a,
                                             int nrows,
                                             int ncols,
                                             float* __restrict__ derivative) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addRepeatAlongRowDerivative<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(repeats, a, nrows, ncols, derivative);
        return cudaGetLastError();
    }

    cudaError_t _repeatAlongCol(cudaStream_t stream,
                                int repeats,
                                int nrows,
                                int ncols,
                                const float* __restrict__ a,
                                float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        repeatAlongCol<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(repeats, nrows, ncols, a, out);
        return cudaGetLastError();
    }

    cudaError_t _addRepeatAlongColDerivative(cudaStream_t stream,
                                             int repeats,
                                             const float* __restrict__ a,
                                             int nrows,
                                             int ncols,
                                             float* __restrict__ derivative) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addRepeatAlongColDerivative<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(repeats, a, nrows, ncols, derivative);
        return cudaGetLastError();
    }

    cudaError_t _batchHorizontalSplit(cudaStream_t stream,
                                      int n,
                                      int nrows,
                                      int xNcols,
                                      int yNcols,
                                      const float* matrices[],
                                      float* xMatrices[],
                                      float* yMatrices[]) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (n * (nrows * xNcols + nrows * yNcols) - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
           batchHorizontalSplit<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(n, nrows, xNcols, yNcols, matrices, xMatrices, yMatrices);
        return cudaGetLastError();
    }


    cudaError_t _batchHorizontalStack(cudaStream_t stream,
                                      int n,
                                      int nrows,
                                      int xNcols,
                                      int yNcols,
                                      const float* xMatrices[],
                                      const float* yMatrices[],
                                      float* outMatrices[]) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (n * (nrows * xNcols + nrows * yNcols) - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
           batchHorizontalStack<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(n, nrows, xNcols, yNcols, xMatrices, yMatrices, outMatrices);
        return cudaGetLastError();
    }


    cudaError_t _maskColumnNumbersRowWise(cudaStream_t stream,
                                          int nrows,
                                          int ncols,
                                          const int* __restrict__ numbers,
                                          float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        maskColumnNumbersRowWise<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, numbers, out);
        return cudaGetLastError();
    }


    cudaError_t _maskZeros(cudaStream_t stream,
                           int nelems,
                           const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        maskZeros<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, out);
        return cudaGetLastError();
    }


    cudaError_t _addMaskZeros(cudaStream_t stream,
                              int nelems,
                              const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addMaskZeros<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, out);
        return cudaGetLastError();
    }


    cudaError_t _dropout(cudaStream_t stream,
                         int nelems,
                         float dropout_prob,
                         const float* __restrict__ data,
                         const float* __restrict__ uniform_data,
                         float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        dropout<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, dropout_prob, data, uniform_data, out);
        return cudaGetLastError();
    }


    cudaError_t _assignMaskedAddition(cudaStream_t stream,
                                      int nelems,
                                      const float* __restrict__ mask,
                                      const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignMaskedAddition<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, mask, a, b, out);
        return cudaGetLastError();
    }


    cudaError_t _assignMaskedAdditionColumnBroadcasted(cudaStream_t stream,
                                                       int nrows,
                                                       int ncols,
                                                       const float* __restrict__ mask,
                                                       const float* __restrict__ a,
                                                       const float* __restrict__ b,
                                                       float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignMaskedAdditionColumnBroadcasted<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, mask, a, b, out);
        return cudaGetLastError();
    }


    cudaError_t _addHprodOneMinusMask(cudaStream_t stream,
                                      int nelems,
                                      const float* __restrict__ mask,
                                      const float* __restrict__ a,
                                      float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addHprodOneMinusMask<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, mask, a, out);
        return cudaGetLastError();
    }


    cudaError_t _addHprodOneMinusMaskColumnBroadcasted(cudaStream_t stream,
                                                       int nrows,
                                                       int ncols,
                                                       const float* __restrict__ mask,
                                                       const float* __restrict__ a,
                                                       float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols- 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addHprodOneMinusMaskColumnBroadcasted<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, mask, a, out);
        return cudaGetLastError();
    }


    cudaError_t _horizontalSliceSplit(cudaStream_t stream,
                                      int n,
                                      int* col_slices,
                                      int nrows,
                                      float** matrices,
                                      float* stacked) {
        size_t float_size = sizeof(float);
        int nelems;
        int offset = 0;

        for (int i = 0; i < n; i++) {
            nelems = (col_slices[i*2+1] - col_slices[i*2]) * nrows;
            offset = col_slices[i*2] * nrows;
            cudaMemcpyAsync(matrices[i], stacked + offset, float_size * nelems, cudaMemcpyDeviceToDevice, stream);
        }
        return cudaGetLastError();
    }

    cudaError_t _horizontalSplit(cudaStream_t stream,
                                 int n,
                                 int* ncols,
                                 int nrows,
                                 float** matrices,
                                 float* stacked) {
        size_t float_size = sizeof(float);
        int nelems;
        int offset = 0;

        for (int i = 0; i < n; i++) {
            nelems = ncols[i] * nrows;
            cudaMemcpyAsync(matrices[i], stacked + offset, float_size * nelems, cudaMemcpyDeviceToDevice, stream);
            offset += nelems;
        }
        return cudaGetLastError();
    }

    cudaError_t _horizontalStack(cudaStream_t stream,
                                 int n,
                                 int* ncols,
                                 int nrows,
                                 float** matrices,
                                 float* stacked) {
        size_t float_size = sizeof(float);
        int nelems;
        int offset = 0;

        for (int i = 0; i < n; i++) {
            nelems = ncols[i] * nrows;
            cudaMemcpyAsync(stacked + offset, matrices[i], float_size * nelems, cudaMemcpyDeviceToDevice, stream);
            offset += nelems;
        }
        return cudaGetLastError();
    }


    cudaError_t _verticalSliceSplit(cudaStream_t stream,
                                    int n,
                                    int* row_slices,
                                    int nrows,
                                    int ncols,
                                    float** matrices,
                                    float* stacked) {
        size_t float_size = sizeof(float);
        float* column_address;
        int offset = 0;
        int k;

        for (int i = 0; i < ncols; i++) {
            for (int j = 0; j < n; j++) {
                k = row_slices[j*2+1] - row_slices[j*2];
                column_address = matrices[j] + k * i;
                offset = nrows * i + row_slices[j*2];
                cudaMemcpyAsync(column_address, stacked + offset, float_size * k, cudaMemcpyDeviceToDevice, stream);
            }
        }

        return cudaGetLastError();
    }

    cudaError_t _verticalSplit(cudaStream_t stream,
                               int n,
                               int* nrows,
                               int ncols,
                               float** matrices,
                               float* stacked) {
        size_t float_size = sizeof(float);
        float* column_address;
        int offset = 0;

        for (int i = 0; i < ncols; i++) {
            for (int j = 0; j < n; j++) {
                column_address = matrices[j] + nrows[j] * i;
                cudaMemcpyAsync(column_address, stacked + offset, float_size * nrows[j], cudaMemcpyDeviceToDevice, stream);
                offset += nrows[j];
            }
        }
        return cudaGetLastError();
    }


    cudaError_t _verticalStack(cudaStream_t stream,
                               int n,
                               int* nrows,
                               int ncols,
                               float** matrices,
                               float* stacked) {
        size_t float_size = sizeof(float);
        float* column_address;
        int offset = 0;

        for (int i = 0; i < ncols; i++) {
            for (int j = 0; j < n; j++) {
                column_address = matrices[j] + nrows[j] * i;
                cudaMemcpyAsync(stacked + offset, column_address, float_size * nrows[j], cudaMemcpyDeviceToDevice, stream);
                offset += nrows[j];
            }
        }

        return cudaGetLastError();
    }


    cudaError_t _sliceColumns(cudaStream_t stream,
                              int nrows,
                              int ncols,
                              const int* __restrict__ embedding_column_indxs,
                              const float* __restrict__ embedding_matrix,
                              float* __restrict__ dense_matrix) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols  - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sliceColumns<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, embedding_column_indxs, embedding_matrix, dense_matrix);
        return cudaGetLastError();
    }


    cudaError_t _sliceColumnsAndTranspose(cudaStream_t stream,
                                          int nrows,
                                          int ncols,
                                          const int* __restrict__ embedding_column_indxs,
                                          const float* __restrict__ embedding_matrix,
                                          float* __restrict__ dense_matrix) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols  - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sliceColumnsAndTranspose<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, embedding_column_indxs, embedding_matrix, dense_matrix);
        return cudaGetLastError();
    }


    cudaError_t _sliceRowsFloat(cudaStream_t stream,
                                int embedding_matrix_nrows,
                                const int* __restrict__ embedding_row_indxs,
                                const float* __restrict__ embedding_matrix,
                                int nrows,
                                int ncols,
                                float* __restrict__ dense_matrix) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols  - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sliceRows<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(embedding_matrix_nrows, embedding_row_indxs, embedding_matrix, nrows, ncols, dense_matrix);
        return cudaGetLastError();
    }


    cudaError_t _sliceRowsInt(cudaStream_t stream,
                              int embedding_matrix_nrows,
                              const int* __restrict__ embedding_row_indxs,
                              const int* __restrict__ embedding_matrix,
                              int nrows,
                              int ncols,
                              int* __restrict__ dense_matrix) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols  - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sliceRows<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(embedding_matrix_nrows, embedding_row_indxs, embedding_matrix, nrows, ncols, dense_matrix);
        return cudaGetLastError();
    }


    cudaError_t _sliceRowsBatch(cudaStream_t stream,
                                const int* embd_rows_indxs,
                                int nrows,
                                int ncols,
                                const float* __restrict__ embd_matrix,
                                int embd_nrows,
                                int embd_ncols,
                                float* __restrict__ dense_matrices[]) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * embd_ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sliceRowsBatch<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(embd_rows_indxs, nrows, ncols, embd_matrix, embd_nrows, embd_ncols, dense_matrices);
        return cudaGetLastError();
    }

    cudaError_t _slicedRowsBatchScaledAdd(cudaStream_t stream,
                                          const int* embd_rows_indxs,
                                          int nrows,
                                          int ncols,
                                          float alpha,
                                          const float* __restrict__ dense_matrices[],
                                          int embd_nrows,
                                          int embd_ncols,
                                          float* __restrict__ embd_matrix) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * embd_ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        slicedRowsBatchScaledAdd<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(embd_rows_indxs, nrows, ncols, alpha, dense_matrices, embd_nrows, embd_ncols, embd_matrix);
        return cudaGetLastError();
    }


    cudaError_t _columnArgmax(cudaStream_t stream,
                              int nrows,
                              int ncols,
                              const float* __restrict__ a,
                              int* __restrict__ indxs) {
        columnArgmax<<<nrows, 32, 32 * sizeof(float), stream>>>(nrows, ncols, a, indxs);
        return cudaGetLastError();
    }


    cudaError_t _hprodSum(cudaStream_t stream,
                          int nrows,
                          int ncols,
                          const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ c) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        fill<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, 0.0, c);
        int nelems = nrows * ncols;
        num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        hprodSum<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, nrows, a, b, c);
        return cudaGetLastError();
    }


    cudaError_t _sumHprod4(cudaStream_t stream,
                           int nelems,
                           const float* __restrict__ a,
                           const float* __restrict__ b,
                           const float* __restrict__ c,
                           const float* __restrict__ d,
                           float* __restrict__ e) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sumHprod<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, d, e);
        return cudaGetLastError();
    }


    cudaError_t _sumHprod5(cudaStream_t stream,
                           int nelems,
                           const float* __restrict__ a,
                           const float* __restrict__ b,
                           const float* __restrict__ c,
                           const float* __restrict__ d,
                           const float* __restrict__ e,
                           float* __restrict__ f) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sumHprod<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, d, e, f);
        return cudaGetLastError();
    }


    cudaError_t _sumHprod11(cudaStream_t stream,
                            int nelems,
                            const float* __restrict__ a,
                            const float* __restrict__ b,
                            const float* __restrict__ c,
                            const float* __restrict__ d,
                            const float* __restrict__ e,
                            const float* __restrict__ f,
                            const float* __restrict__ g,
                            const float* __restrict__ h,
                            const float* __restrict__ i,
                            const float* __restrict__ j,
                            const float* __restrict__ k,
                            float* __restrict__ l) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sumHprod<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, d, e, f, g, h, i, j, k, l);
        return cudaGetLastError();
    }


    cudaError_t _hadamardProduct2(cudaStream_t stream,
                                  int nelems,
                                  const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        hadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c);
        return cudaGetLastError();
    }


    cudaError_t _hadamardProduct3(cudaStream_t stream,
                                  int nelems,
                                  const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  const float* __restrict__ c,
                                  float* __restrict__ d) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        hadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, d);
        return cudaGetLastError();
    }


    cudaError_t _addHadamardProduct2(cudaStream_t stream,
                                     int nelems,
                                     const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float alpha,
                                     float* __restrict__ c) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addHadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, alpha, c);
        return cudaGetLastError();
    }


    cudaError_t _addHadamardProduct3(cudaStream_t stream,
                                     int nelems,
                                     const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     const float* __restrict__ c,
                                     float alpha,
                                     float* __restrict__ d) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addHadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, alpha, d);
        return cudaGetLastError();
    }


    cudaError_t _addScaledHadamardProduct(cudaStream_t stream,
                                          int nelems,
                                          const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float alpha,
                                          float beta,
                                          float* __restrict__ c) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addScaledHadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, alpha, beta, c);
        return cudaGetLastError();
    }


    cudaError_t _addScaledColumnsSlice(cudaStream_t stream,
                                       int nrows,
                                       int ncols,
                                       float alpha,
                                       const float* __restrict__ dense_matrix,
                                       const int* __restrict__ embedding_column_indxs,
                                       float* __restrict__ embedding_matrix) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addScaledColumnsSlice<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, alpha, dense_matrix, embedding_column_indxs, embedding_matrix);
        return cudaGetLastError();
    }


    cudaError_t _addScaledRowsSlice(cudaStream_t stream,
                                    int nrows,
                                    int ncols,
                                    float alpha,
                                    const float* __restrict__ dense_matrix,
                                    const int* __restrict__ embedding_row_indxs,
                                    int embd_nrows,
                                    float* __restrict__ embedding_matrix) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addScaledRowsSlice<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, alpha, dense_matrix, embedding_row_indxs, embd_nrows, embedding_matrix);
        return cudaGetLastError();
    }


    cudaError_t _add_sum(cudaStream_t stream,
                         int nelems,
                         const float* matrices[],
                         int n,
                         float* __restrict__ s) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addSum<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, matrices, n, s);
        return cudaGetLastError();
    }


    cudaError_t _assign_sum(cudaStream_t stream,
                            int nelems,
                            const float* matrices[],
                            int n,
                            float* __restrict__ s) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignSum<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, matrices, n, s);
        return cudaGetLastError();
    }


    cudaError_t _scale(cudaStream_t stream,
                       int nelems,
                       float alpha,
                       const float* __restrict__ data,
                       float* __restrict__ out_data) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        scale<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, alpha, out_data);
        return cudaGetLastError();
    }


    cudaError_t _fill(cudaStream_t stream,
                      int nelems,
                      float value,
                      float* __restrict__ out_data) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        fill<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, value, out_data);
        return cudaGetLastError();
    }

    cudaError_t _maskedFill(cudaStream_t stream,
                            int nelems,
                            float value,
                            const float* __restrict__ mask,
                            float true_value,
                            float* __restrict__ out_data) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        maskedFill<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, value, mask, true_value, out_data);
        return cudaGetLastError();
    }

    cudaError_t _matrixVectorRowAddition(cudaStream_t stream,
                                         int nrows,
                                         int ncols,
                                         const float* __restrict__ matrix,
                                         float alpha,
                                         const float* __restrict__ vector,
                                         float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        matrixVectorRowAddition<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, matrix, alpha, vector, out);
        return cudaGetLastError();
    }


    cudaError_t _assignSequentialMeanPooling(cudaStream_t stream,
                                             int nrows,
                                             int ncols,
                                             const float* matrices[],
                                             int n,
                                             float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignSequentialMeanPooling<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, matrices, n, out);
        return cudaGetLastError();
    }

    cudaError_t _assignSequentialSumPooling(cudaStream_t stream,
                                            int nrows,
                                            int ncols,
                                            const float* matrices[],
                                            int n,
                                            float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignSequentialSumPooling<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, matrices, n, out);
        return cudaGetLastError();
    }

    cudaError_t _assignSequentialWeightedSum(cudaStream_t stream,
                                            int nrows,
                                            int ncols,
                                            const float* matrices[],
                                            const float* __restrict__ weights,
                                            int n,
                                            float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignSequentialWeightedSum<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, matrices, weights, n, out);
        return cudaGetLastError();
    }

    cudaError_t _sequentiallyTile(cudaStream_t stream,
                                  int nelems,
                                  const float* __restrict__ a,
                                  float* matrices[],
                                  int n) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sequentiallyTile<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, matrices, n);
        return cudaGetLastError();
    }


    cudaError_t _assignScaledAddition(cudaStream_t stream,
                                      int nelems,
                                      float alpha,
                                      const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignScaledAddition<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, alpha, a, b, out);
        return cudaGetLastError();
    }


    cudaError_t _assignScaledSubtraction(cudaStream_t stream,
                                         int nelems,
                                         float alpha,
                                         const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignScaledSubtraction<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, alpha, a, b, out);
        return cudaGetLastError();
    }


    cudaError_t _addScaledSubtraction(cudaStream_t stream,
                                      int nelems,
                                      float alpha,
                                      const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addScaledSubtraction<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, alpha, a, b, out);
        return cudaGetLastError();
    }


    cudaError_t _softmaxCeDerivative(cudaStream_t stream,
                                     int batchSize,
                                     int numClasses,
                                     const float* __restrict__ probs,
                                     const int* __restrict__ targetClasses,
                                     float* __restrict__ derivatives) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (batchSize * numClasses - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        softmaxCeDerivative<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(batchSize, numClasses, probs, targetClasses, derivatives);
        return cudaGetLastError();
    }


    cudaError_t _addSoftmaxCeDerivative(cudaStream_t stream,
                                        int batchSize,
                                        int numClasses,
                                        const float* __restrict__ probs,
                                        const int* __restrict__ targetClasses,
                                        float* __restrict__ derivatives) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (batchSize * numClasses - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addSoftmaxCeDerivative<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(batchSize, numClasses, probs, targetClasses, derivatives);
        return cudaGetLastError();
    }


    cudaError_t _matrixVectorColumnHprod(cudaStream_t stream,
                                         int nrows,
                                         int ncols,
                                         const float* __restrict__ matrix,
                                         const float* __restrict__ vector,
                                         float* out) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        matrixVectorColumnHprod<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, matrix, vector, out);
        return cudaGetLastError();
    }
}