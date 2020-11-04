#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
//��
#include <cudnn.h>
#include <stdlib.h>
#include <iostream>
//��-
using namespace std;
#define MAX_THREADS_NUM 512
#define MAX_BLOCKS_NUM 4096
#define BLOCK_NUM(count) min(((count + MAX_THREADS_NUM - 1) / MAX_THREADS_NUM), MAX_BLOCKS_NUM)
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)

//��
#define CUDNN_CALL(f) { \
cudnnStatus_t err = (f); \
if (err != CUDNN_STATUS_SUCCESS) {\
    \
        std::cout << "    Error occurred: " << err << std::endl; \
        std::exit(1); \
} \
}
//��-
__global__ void matrix_array_set_kernel(int count,
                                        float *arr,
                                        float value) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    arr[index] = value;
  }
}

__global__ void matrix_broadcast_to_kernel(int inputCount, float* inputArr,
                                           int outputCount, float* outputArr) {
  CUDA_1D_KERNEL_LOOP(index, outputCount) {
      outputArr[index] = inputArr[index % inputCount];
  }
}

__global__ void matrix_reduce_sum_axis_zero_kernel(float* inputArr,
                                                   int outputCount, float* outputArr,
                                                   int zeroDim) {
      CUDA_1D_KERNEL_LOOP(index, outputCount) {
          float sum = 0;
          for (int i = 0; i < zeroDim; ++i) {
              sum += inputArr[index + i * outputCount];
          }
          outputArr[index] = sum;
      }
}

__global__ void matrix_elementwise_add_kernel(float* matAData, float* matBData,
                                              float* outputData, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputData[index] = matAData[index] + matBData[index];
    }
}

__global__ void matrix_elementwise_add_by_const_kernel(float* inputArr, float val,
                                                       float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index] + val;
    }
}

__global__ void matrix_elementwise_multiply_kernel(float* matAData, float* matBData,
                                                   float* outputData, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputData[index] = matAData[index] * matBData[index];
    }
}

__global__ void matrix_elementwise_multipy_by_const_kernel(float* inputArr, float val,
                                                           float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index] * val;
    }
}

__global__ void matrix_relu_kernel(float* inputArr, float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index];
        if (inputArr[index] < 0) {
            outputArr[index] = 0.f;
        }
    }
}

__global__ void matrix_relu_gradient_kernel(const float* inputArr, const float* gradArr,
                                            float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index] > 0 ? gradArr[index] : 0;
    }
}

__global__ void matrix_softmax_kernel(int nRow, int nCol, float* inputArr, float* outputArr) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= nRow) return;

    float* input = inputArr + y * nCol;
    float* output = outputArr + y * nCol;

    float maxval = *input;
    for (int i = 1; i < nCol; ++i) {
        maxval = max(input[i], maxval);
    }
    float sum = 0;
    for (int i = 0; i < nCol; ++i) {
        sum += expf(input[i] - maxval);
    }
    for (int i = 0; i < nCol; ++i) {
        output[i] = expf(input[i] - maxval) / sum;
    }
}

/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) {
  int count = 1;
  for (int i = 0; i < arr->ndim; ++i) {
    count *= arr->shape[i];
  }
  float *arr_data = (float *)arr->data;
  matrix_array_set_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
    count, arr_data, value);
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim + 1 == output->ndim);
  int inputCount = 1, outputCount = output->shape[0];
  for (int i = 0; i < input->ndim; ++i) {
      assert(input->shape[i] == output->shape[i + 1]);
      inputCount *= input->shape[i];
      outputCount *= output->shape[i + 1];
  }
  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_broadcast_to_kernel<<<BLOCK_NUM(outputCount), MAX_THREADS_NUM>>>(
    inputCount, inputArr, outputCount, outputArr);
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim == output->ndim + 1);
  int zeroDim = input->shape[0], outputCount = 1;
    for (int i = 0; i < output->ndim; ++i) {
        assert(input->shape[i+1] == output->shape[i]);
        outputCount *= output->shape[i];
    }
  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_reduce_sum_axis_zero_kernel<<<BLOCK_NUM(outputCount), MAX_THREADS_NUM>>>(
          inputArr, outputCount, outputArr, zeroDim);
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  assert(matA->ndim == output->ndim);
  assert(matB->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < matA->ndim; ++i) {
    assert(matA->shape[i] == output->shape[i]);
    assert(matB->shape[i] == output->shape[i]);
    count *= matA->shape[i];
  }
  float* matAData = (float*) matA->data;
  float* matBData = (float*) matB->data;
  float* outputData = (float*) output->data;
  matrix_elementwise_add_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
          matAData, matBData, outputData, count);
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  assert(input->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    count *= input->shape[i];
  }
  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_elementwise_add_by_const_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
          inputArr, val, outputArr, count);
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  assert(matA->ndim == output->ndim);
  assert(matB->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < matA->ndim; ++i) {
    assert(matA->shape[i] == output->shape[i]);
    assert(matB->shape[i] == output->shape[i]);
    count *= matA->shape[i];
  }
  float* matAData = (float*) matA->data;
  float* matBData = (float*) matB->data;
  float* outputData = (float*) output->data;
  matrix_elementwise_multiply_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
          matAData, matBData, outputData, count);
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  assert(input->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    count *= input->shape[i];
  }
  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_elementwise_multipy_by_const_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
          inputArr, val, outputArr, count);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  // Hint: use cublas
  // cublas assume matrix is column major
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(matC->ndim == 2);
  assert(matA->shape[transposeA ? 0 : 1] == matB->shape[transposeB ? 1 : 0]);
  assert(matA->shape[transposeA ? 1 : 0] == matC->shape[0]);
  assert(matB->shape[transposeB ? 0 : 1] == matC->shape[1]);

  cublasHandle_t handle;
  cublasCreate(&handle);
  const float* matAData = (const float*) matA->data;
  const float* matBData = (const float*) matB->data;
  float* matCData = (float*) matC->data;
  float alpha = 1, beta = 0;

  cublasSgemm(handle,
              (transposeB ? CUBLAS_OP_T : CUBLAS_OP_N),
              (transposeA ? CUBLAS_OP_T : CUBLAS_OP_N),
              (transposeB ? matB->shape[0] : matB->shape[1]),
              (transposeA ? matA->shape[1] : matA->shape[0]),
              (transposeB ? matB->shape[1] : matB->shape[0]),
              &alpha,
              matBData, matB->shape[1],
matAData, matA->shape[1],
& beta,
matCData, (transposeB ? matB->shape[0] : matB->shape[1]));

return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == output->ndim);
    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
        assert(input->shape[i] == output->shape[i]);
        count *= input->shape[i];
    }
    float* inputArr = (float*)input->data;
    float* outputArr = (float*)output->data;
    matrix_relu_kernel << <BLOCK_NUM(count), MAX_THREADS_NUM >> > (
        inputArr, outputArr, count);
    return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
    DLArrayHandle output) {
    assert(input->ndim == in_grad->ndim);
    assert(input->ndim == output->ndim);
    int count = 1;
    for (int i = 0; i < input->ndim; ++i) {
        assert(input->shape[i] == in_grad->shape[i]);
        assert(input->shape[i] == output->shape[i]);
        count *= input->shape[i];
    }
    const float* inputArr = (const float*)input->data;
    const float* gradArr = (const float*)in_grad->data;
    float* outputArr = (float*)output->data;
    matrix_relu_gradient_kernel << <BLOCK_NUM(count), MAX_THREADS_NUM >> > (
        inputArr, gradArr, outputArr, count);
    return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    assert(input->shape[0] == output->shape[0]);
    assert(input->shape[1] == output->shape[1]);

    int nRow = input->shape[0];
    int nCol = input->shape[1];

    dim3 block(MAX_THREADS_NUM);
    dim3 grid((nRow + block.x - 1) / block.x);

    float* inputArr = (float*)input->data;
    float* outputArr = (float*)output->data;

    matrix_softmax_kernel << <grid, block >> > (nRow, nCol, inputArr, outputArr);

    return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
    const DLArrayHandle input_b,
    DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(input_b->ndim == 2);
    assert(output->ndim == 1);
    assert(input_a->shape[0] == input_b->shape[0] &&
        input_a->shape[1] == input_b->shape[1]);
    int nrow = input_a->shape[0];
    // Maximum x- or y-dimension of a block = 1024
    // But we need 'nrow' shared memory, and max shared memory is 48KB.
    // Conservatively allow max 16KB shared memory.
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float* input_data_a = (const float*)input_a->data;
    const float* input_data_b = (const float*)input_b->data;
    float* output_data = (float*)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    }
    else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_softmax_cross_entropy_kernel << <1, threads, nrow * sizeof(float) >> > (
        nrow, ncol, input_data_a, input_data_b, output_data);
    return 0;
}




//3ά
int DLGpuConvolution1DForward(const DLArrayHandle input,
    const DLArrayHandle filter,
    DLArrayHandle output,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int v          /*filter stride */) {

    cout<<dataformat<<endl;
    cout<<padding<<endl;
    assert(input->ndim == 3);
    assert(filter->ndim == 3);



    int input_n = input->shape[0];
    int input_c = input->shape[2];
    int input_h = 1;
    int input_w = input->shape[1];

    int filter_n = filter->shape[0];
    int filter_c = filter->shape[2];
    int filter_h = 1;
    int filter_w = filter->shape[1];

    if (dataformat == 0) {
        input_n = input->shape[0];
        input_c = input->shape[1];
        input_h = 1;
        input_w = input->shape[2];

        filter_n = filter->shape[0];
        filter_c = filter->shape[1];
        filter_h = 1;
        filter_w = filter->shape[2];
    }

    int out_n;
    int out_c;
    int out_h;
    int out_w;

    int pad_h = 0;
    int pad_w = 0;

    int u = 1;

    if (padding == 1) {
        pad_w = filter_w / 2;
    }


    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));
     printf("0\n");
    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));


    //������
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT,
        dataformat,
        filter_n,
        filter_c,
        filter_h,
        filter_w));


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor,
        pad_h, pad_w, // zero-padding
        u, v, // stride
        1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //����output��4��ά��
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        &out_n,
        &out_c,
        &out_h,
        &out_w));

    if (dataformat == 0) {
        assert(output->shape[0] == out_n);
        assert(output->shape[1] == out_c);
        assert(1 == out_h);
        assert(output->shape[2] == out_w);
    }
    else {
        assert(output->shape[0] == out_n);
        assert(output->shape[2] == out_c);
        assert(1 == out_h);
        assert(output->shape[1] == out_w);
    }



    //output��Ϣ
    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        out_n,
        out_c,
        out_h,
        out_w));

    //�������㷨
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo));

    //׼����������Ŀռ�
    size_t workspace_size = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        algo,
        &workspace_size));
    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    printf("6\n");
    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(handle,
        &alpha, //x*w����
        input_descriptor,
        input->data,
        filter_descriptor,
        filter->data,
        conv_descriptor,
        algo,
        workspace,
        workspace_size,
        &beta, //y����,y�������ݽ������ţ�
        output_descriptor,
        output->data));
    //�ڴ�
    cudaFree(workspace);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));

    return 0;

}


int DLGpuConvolution1DForwardGetOutShape(const int* input_shapes,
    const int* filter_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int v          /*filter stride */) {

    cout << dataformat << endl;
    cout << padding << endl;


    int input_n = input_shapes[0];
    int input_c = input_shapes[2];
    int input_h = 1;
    int input_w = input_shapes[1];

    int filter_n = filter_shapes[0];
    int filter_c = filter_shapes[2];
    int filter_h = 1;
    int filter_w = filter_shapes[1];

    if (dataformat == 0) {
        input_n = input_shapes[0];
        input_c = input_shapes[1];
        input_h = 1;
        input_w = input_shapes[2];

        filter_n = filter_shapes[0];
        filter_c = filter_shapes[1];
        filter_h = 1;
        filter_w = filter_shapes[2];
    }



    int out_n;
    int out_c;
    int out_h;
    int out_w;

    int pad_h = 0;
    int pad_w = 0;

    int u = 1;

    if (padding == 1) {
        pad_w = filter_w / 2;
    }


    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));

    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));


    //������
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT,
        dataformat,
        filter_n,
        filter_c,
        filter_h,
        filter_w));


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor,
        pad_h, pad_w, // zero-padding
        u, v, // stride
        1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //����output��4��ά��
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        &out_n,
        &out_c,
        &out_h,
        &out_w));



    if (dataformat == 0) {
        output_shapes[0] = out_n;
        output_shapes[1] = out_c;
        output_shapes[2] = out_w;
    }
    else {
        output_shapes[0] = out_n;
        output_shapes[1] = out_w;
        output_shapes[2] = out_c;
    }



    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));

    return 0;

}


int DLGpuConvolution1DBackward(const DLArrayHandle input,
    const DLArrayHandle doutput,
    const DLArrayHandle filter,
    DLArrayHandle dfilter,
    DLArrayHandle dinput,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int v          /*filter stride */) {

    assert(input->ndim == 3);
    assert(filter->ndim == 3);


     int input_n = input->shape[0];
     int input_c = input->shape[2];
     int input_h = 1;
     int input_w = input->shape[1];

     int filter_n = filter->shape[0];
     int filter_c = filter->shape[2];
     int filter_h = 1;
     int filter_w = filter->shape[1];


    if (dataformat == 0) {
        input_n = input->shape[0];
        input_c = input->shape[1];
        input_h = 1;
        input_w = input->shape[2];

        filter_n = filter->shape[0];
        filter_c = filter->shape[1];
        filter_h = 1;
        filter_w = filter->shape[2];
    }

  

    int out_n;
    int out_c;
    int out_h;
    int out_w;

    int pad_h = 0;
    int pad_w = 0;

    int u = 1;

    if (padding == 1) {
        pad_w = filter_w / 2;
    }


    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));

    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));


    //������
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT,
        dataformat,
        filter_n,
        filter_c,
        filter_h,
        filter_w));


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor,
        pad_h, pad_w, // zero-padding
        u, v, // stride
        1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //����output��4��ά��
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        &out_n,
        &out_c,
        &out_h,
        &out_w));

  

    if (dataformat == 0) {
        assert(doutput->shape[0] == out_n);
        assert(doutput->shape[1] == out_c);
        assert(1 == out_h);
        assert(doutput->shape[2] == out_w);
    }
    else {
        assert(doutput->shape[0] == out_n);
        assert(doutput->shape[2] == out_c);
        assert(1 == out_h);
        assert(doutput->shape[1] == out_w);
    }


    //output��Ϣ
    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        out_n,
        out_c,
        out_h,
        out_w));

    //�������㷨
    cudnnConvolutionBwdFilterAlgo_t  algo1;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
        input_descriptor,
        output_descriptor,
        conv_descriptor,
        filter_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        &algo1));

    cudnnConvolutionBwdDataAlgo_t algo2;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle,
        filter_descriptor,
        output_descriptor,
        conv_descriptor,
        input_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo2));

    //׼����������Ŀռ�


    size_t workspace_size1= 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
        input_descriptor,
        output_descriptor,
        conv_descriptor,
        filter_descriptor,
        algo1,
        &workspace_size1));
    void* workspace1= nullptr;
    cudaMalloc(&workspace1, workspace_size1);

    size_t workspace_size2 = 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
        filter_descriptor,
        output_descriptor,
        conv_descriptor,
        input_descriptor,
        algo2,
        &workspace_size2));
    void* workspace2 = nullptr;
    cudaMalloc(&workspace2, workspace_size2);



    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
        &alpha, //x*w����
        input_descriptor,
        input->data,
        output_descriptor,
        doutput->data,
        conv_descriptor,
        algo1,
        workspace1,
        workspace_size1,
        &beta, //y����,y�������ݽ������ţ�
        filter_descriptor,
        dfilter->data));



    CUDNN_CALL(cudnnConvolutionBackwardData(handle,
        &alpha, //x*w����
        filter_descriptor,
        filter->data,
        output_descriptor,
        doutput->data,
        conv_descriptor,
        algo2,
        workspace2,
        workspace_size2,
        &beta, //y����,y�������ݽ������ţ�
        input_descriptor,
        dinput->data));


    //�ڴ�
    cudaFree(workspace1);
    cudaFree(workspace2);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));

    return 0;

}





//4ά
int DLGpuConvolution2DForward(const DLArrayHandle input,
    const DLArrayHandle filter,
    DLArrayHandle output,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int u,          /* vertical filter stride */
    const int v          /* horizontal filter stride */){

    assert(input->ndim == 4);
    assert(filter->ndim == 4);


    int input_n = input->shape[0];
    int input_c = input->shape[3];
    int input_h = input->shape[1];
    int input_w = input->shape[2];

    int filter_n = filter->shape[0];
    int filter_c = filter->shape[3];
    int filter_h = filter->shape[1];
    int filter_w = filter->shape[2];


    if (dataformat == 0) {
        input_n = input->shape[0];
        input_c = input->shape[1];
        input_h = input->shape[2];
        input_w = input->shape[3];

        filter_n = filter->shape[0];
        filter_c = filter->shape[1];
        filter_h = filter->shape[2];
        filter_w = filter->shape[3];
    }



    int out_n;
    int out_c;
    int out_h;
    int out_w;

    int pad_h = 0;
    int pad_w = 0;

    if (padding == 1) {
        pad_h = filter_h / 2;
        pad_w = filter_w / 2;
    }


    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));

    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n, 
        input_c, 
        input_h, 
        input_w));


    //������
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT,
        dataformat,
        filter_n, 
        filter_c, 
        filter_h, 
        filter_w));


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor,
        pad_h, pad_w, // zero-padding
        u, v, // stride
        1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //����output��4��ά��
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        &out_n,
        &out_c,
        &out_h,
        &out_w));

   

    if (dataformat == 0) {
        assert(output->shape[0] == out_n);
        assert(output->shape[1] == out_c);
        assert(output->shape[2] == out_h);
        assert(output->shape[3] == out_w);
    }
    else {
        assert(output->shape[0] == out_n);
        assert(output->shape[3] == out_c);
        assert(output->shape[1] == out_h);
        assert(output->shape[2] == out_w);
    }


    //output��Ϣ
    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        out_n,
        out_c,
        out_h,
        out_w));

    //�������㷨
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo));

    //׼����������Ŀռ�
    size_t workspace_size = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        algo,
        &workspace_size));
    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);


    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(handle,
        &alpha, //x*w����
        input_descriptor,
        input->data,
        filter_descriptor,
        filter->data,
        conv_descriptor,
        algo,
        workspace,
        workspace_size,
        &beta, //y����,y�������ݽ������ţ�
        output_descriptor,
        output->data));

    //�ڴ�
    cudaFree(workspace);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));
    return 0;
}


int DLGpuConvolution2DBackward(const DLArrayHandle input,
    const DLArrayHandle doutput,
    const DLArrayHandle filter,
    DLArrayHandle dfilter,
    DLArrayHandle dinput,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int u,          /* vertical filter stride */
    const int v          /* horizontal filter stride */){

    assert(input->ndim == 4);
    assert(filter->ndim == 4);

    int input_n = input->shape[0];
    int input_c = input->shape[3];
    int input_h = input->shape[1];
    int input_w = input->shape[2];

    int filter_n = filter->shape[0];
    int filter_c = filter->shape[3];
    int filter_h = filter->shape[1];
    int filter_w = filter->shape[2];


    if (dataformat == 0) {
        input_n = input->shape[0];
        input_c = input->shape[1];
        input_h = input->shape[2];
        input_w = input->shape[3];

        filter_n = filter->shape[0];
        filter_c = filter->shape[1];
        filter_h = filter->shape[2];
        filter_w = filter->shape[3];
    }





    int out_n;
    int out_c;
    int out_h;
    int out_w;

    int pad_h = 0;
    int pad_w = 0;

    if (padding == 1) {
        pad_h = filter_h / 2;
        pad_w = filter_w / 2;
    }


    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));

    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));


    //������
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT,
        dataformat,
        filter_n,
        filter_c,
        filter_h,
        filter_w));


    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor,
        pad_h, pad_w, // zero-padding
        u, v, // stride
        1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //����output��4��ά��
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        &out_n,
        &out_c,
        &out_h,
        &out_w));

    assert(doutput->shape[0] == out_n);
    assert(doutput->shape[1] == out_c);
    assert(doutput->shape[2] == out_h);
    assert(doutput->shape[3] == out_w);




    //output��Ϣ
    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        out_n,
        out_c,
        out_h,
        out_w));

    cudnnConvolutionBwdFilterAlgo_t  algo1;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
        input_descriptor,
        output_descriptor,
        conv_descriptor,
        filter_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        &algo1));

    cudnnConvolutionBwdDataAlgo_t algo2;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle,
        filter_descriptor,
        output_descriptor,
        conv_descriptor,
        input_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo2));

    //׼����������Ŀռ�


    size_t workspace_size1= 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
        input_descriptor,
        output_descriptor,
        conv_descriptor,
        filter_descriptor,
        algo1,
        &workspace_size1));
    void* workspace1= nullptr;
    cudaMalloc(&workspace1, workspace_size1);

    size_t workspace_size2 = 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
        filter_descriptor,
        output_descriptor,
        conv_descriptor,
        input_descriptor,
        algo2,
        &workspace_size2));
    void* workspace2 = nullptr;
    cudaMalloc(&workspace2, workspace_size2);



    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
        &alpha, //x*w����
        input_descriptor,
        input->data,
        output_descriptor,
        doutput->data,
        conv_descriptor,
        algo1,
        workspace1,
        workspace_size1,
        &beta, //y����,y�������ݽ������ţ�
        filter_descriptor,
        dfilter->data));



    CUDNN_CALL(cudnnConvolutionBackwardData(handle,
        &alpha, //x*w����
        filter_descriptor,
        filter->data,
        output_descriptor,
        doutput->data,
        conv_descriptor,
        algo2,
        workspace2,
        workspace_size2,
        &beta, //y����,y�������ݽ������ţ�
        input_descriptor,
        dinput->data));


    //�ڴ�
    cudaFree(workspace1);
    cudaFree(workspace2);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));
    return 0;
}

//5ά
int DLGpuConvolution3DForward(const DLArrayHandle input,
    const DLArrayHandle filter,
    DLArrayHandle output,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int s1,          
    const int s2,     
    const int s3) {

    assert(input->ndim == 5);
    assert(filter->ndim == 5);

    int* input_shape, * output_shape, * filter_shape, * inputstrides,*outputstrides;

    int* padA, * filterStrideA, * dilationA;

    input_shape = (int*)malloc(sizeof(int) * 5);
    filter_shape = (int*)malloc(sizeof(int) * 5);
    output_shape = (int*)malloc(sizeof(int) * 5);
    inputstrides = (int*)malloc(sizeof(int) * 5);
    outputstrides = (int*)malloc(sizeof(int) * 5);
    padA = (int*)malloc(sizeof(int) * 3);
    filterStrideA = (int*)malloc(sizeof(int) * 3);
    dilationA = (int*)malloc(sizeof(int) * 3);

    for (int i=0;i<5;i++)
    {
        input_shape[i]=input->shape[i];
        filter_shape[i]=filter->shape[i];
    }


    for (int i = 0; i < 3; i++) {
        padA[i] = 0;
        dilationA[i] = 1;
    }

    if (padding == 1) {
        for (int i = 0; i < 3; i++) {
            padA[i] = filter_shape[i+2]/2;
            
        }
    }


    filterStrideA[0] = s1;
    filterStrideA[1] = s2;
    filterStrideA[2] = s3;


    inputstrides[4] = 1;
    for (int i = 1; i < 5; i++) {
        inputstrides[4 - i] = inputstrides[5 - i] * input_shape[5 - i];
    }



     cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));


    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        input_shape));

    //�˺���
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilterNdDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT,
        dataformat,
        5,
        filter_shape));

    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(conv_descriptor,
        3,
        padA,
        filterStrideA,
        dilationA,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //output��shape
    CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        5,
        output_shape));


    assert(output->shape[0] == output_shape[0]);
    assert(output->shape[1] == output_shape[1]);
    assert(output->shape[2] == output_shape[2]);
    assert(output->shape[3] == output_shape[3]);
    assert(output->shape[4] == output_shape[4]);


    outputstrides[4] = 1;
    for (int i = 1; i < 5; i++) {
        outputstrides[4 - i] = outputstrides[5 - i] * output_shape[5 - i];
    }


    //output
    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        output_shape));

    //�������㷨
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo));

    //׼����������Ŀռ�
    size_t workspace_size = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        algo,
        &workspace_size));
    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);


    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(handle,
        &alpha, //x*w����
        input_descriptor,
        input->data,
        filter_descriptor,
        filter->data,
        conv_descriptor,
        algo,
        workspace,
        workspace_size,
        &beta, //y����,y�������ݽ������ţ�
        output_descriptor,
        output->data));

    //�ڴ�
    cudaFree(workspace);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));


   return 0;




}

int DLGpuConvolution3DBackward(const DLArrayHandle input,
    const DLArrayHandle doutput,
    const DLArrayHandle filter,
    DLArrayHandle dfilter,
    DLArrayHandle dinput,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int s1,
    const int s2,
    const int s3) {

    assert(input->ndim == 5);
    assert(filter->ndim == 5);

    int* input_shape, * output_shape, * filter_shape, * inputstrides,*outputstrides;

    int* padA, * filterStrideA, * dilationA;

    input_shape = (int*)malloc(sizeof(int) * 5);
    filter_shape = (int*)malloc(sizeof(int) * 5);
    output_shape = (int*)malloc(sizeof(int) * 5);
    inputstrides = (int*)malloc(sizeof(int) * 5);
    outputstrides = (int*)malloc(sizeof(int) * 5);
    padA = (int*)malloc(sizeof(int) * 3);
    filterStrideA = (int*)malloc(sizeof(int) * 3);
    dilationA = (int*)malloc(sizeof(int) * 3);

    for (int i=0;i<5;i++)
    {
        input_shape[i]=input->shape[i];
        filter_shape[i]=filter->shape[i];
    }


    for (int i = 0; i < 3; i++) {
        padA[i] = 0;
        dilationA[i] = 1;
    }

    if (padding == 1) {
        for (int i = 0; i < 3; i++) {
            padA[i] = filter_shape[i+2]/2;

        }
    }


    filterStrideA[0] = s1;
    filterStrideA[1] = s2;
    filterStrideA[2] = s3;


    inputstrides[4] = 1;
    for (int i = 1; i < 5; i++) {
        inputstrides[4 - i] = inputstrides[5 - i] * input_shape[5 - i];
    }



     cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));


    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        input_shape));

    //�˺���
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilterNdDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT,
        dataformat,
        5,
        filter_shape));

    //��ķ�ʽ��������padding
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(conv_descriptor,
        3,
        padA,
        filterStrideA,
        dilationA,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //output��shape
    CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        5,
        output_shape));


    assert(doutput->shape[0] == output_shape[0]);
    assert(doutput->shape[1] == output_shape[1]);
    assert(doutput->shape[2] == output_shape[2]);
    assert(doutput->shape[3] == output_shape[3]);
    assert(doutput->shape[4] == output_shape[4]);


    outputstrides[4] = 1;
    for (int i = 1; i < 5; i++) {
        outputstrides[4 - i] = outputstrides[5 - i] * output_shape[5 - i];
    }


    //output
    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        output_shape));

       cudnnConvolutionBwdFilterAlgo_t  algo1;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
        input_descriptor,
        output_descriptor,
        conv_descriptor,
        filter_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        &algo1));

    cudnnConvolutionBwdDataAlgo_t algo2;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle,
        filter_descriptor,
        output_descriptor,
        conv_descriptor,
        input_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo2));

    //׼����������Ŀռ�


    size_t workspace_size1= 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
        input_descriptor,
        output_descriptor,
        conv_descriptor,
        filter_descriptor,
        algo1,
        &workspace_size1));
    void* workspace1= nullptr;
    cudaMalloc(&workspace1, workspace_size1);

    size_t workspace_size2 = 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
        filter_descriptor,
        output_descriptor,
        conv_descriptor,
        input_descriptor,
        algo2,
        &workspace_size2));
    void* workspace2 = nullptr;
    cudaMalloc(&workspace2, workspace_size2);



    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardFilter(handle,
        &alpha, //x*w����
        input_descriptor,
        input->data,
        output_descriptor,
        doutput->data,
        conv_descriptor,
        algo1,
        workspace1,
        workspace_size1,
        &beta, //y����,y�������ݽ������ţ�
        filter_descriptor,
        dfilter->data));



    CUDNN_CALL(cudnnConvolutionBackwardData(handle,
        &alpha, //x*w����
        filter_descriptor,
        filter->data,
        output_descriptor,
        doutput->data,
        conv_descriptor,
        algo2,
        workspace2,
        workspace_size2,
        &beta, //y����,y�������ݽ������ţ�
        input_descriptor,
        dinput->data));


    //�ڴ�
    cudaFree(workspace1);
    cudaFree(workspace2);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));
    return 0;
}



int DLGpuPooling1DForward(const DLArrayHandle input,
    DLArrayHandle output,
    cudnnTensorFormat_t dataformat,
    const int padding_w,
    const int v,
    const int filter_w) {

    int padding_h = 0;
    int u = 1;
    int filter_h = 1;


    int input_n = input->shape[0];
    int input_c = input->shape[2];
    int input_h = 1;
    int input_w = input->shape[1];



    if (dataformat == 0) {
        input_n = input->shape[0];
        input_c = input->shape[1];
        input_h = 1;
        input_w = input->shape[2];
    }




    int output_n;
    int output_c;
    int output_h;
    int output_w;


    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));


    cudnnPoolingDescriptor_t pool_descriptor;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_descriptor));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_descriptor,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        filter_h, filter_w,
        padding_h, padding_w, // zero-padding
        u, v // stride
    ));


    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));



    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(
        pool_descriptor,
        input_descriptor,
        &output_n,
        &output_c,
        &output_h,
        &output_w));

    if (dataformat == 0) {
        assert(output->shape[0] == output_n);
        assert(output->shape[1] == output_c);
        assert(1 == output_h);
        assert(output->shape[2] == output_w);
    }
    else {
        assert(output->shape[0] == output_n);
        assert(output->shape[2] == output_c);
        assert(1 == output_h);
        assert(output->shape[1] == output_w);
    }
    

    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        output_n,
        output_c,
        output_h,
        output_w));


    auto alpha = 1.0f, beta = 1.0f;
    CUDNN_CALL(cudnnPoolingForward(handle,
        pool_descriptor,
        &alpha,
        input_descriptor,
        input->data,
        &beta,
        output_descriptor,
        output->data));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));
    return 0;
}

int DLGpuPooling1DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
    cudnnTensorFormat_t dataformat,
    const int padding_w,
    const int v,
    const int filter_w) {

    int padding_h = 0;
    int u = 1;
    int filter_h = 1;


    int input_n = input->shape[0];
    int input_c = input->shape[2];
    int input_h = 1;
    int input_w = input->shape[1];

    if (dataformat == 0) {
        input_n = input->shape[0];
        input_c = input->shape[1];
        input_h = 1;
        input_w = input->shape[2];
    }


    int output_n;
    int output_c;
    int output_h;
    int output_w;


    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));


    cudnnPoolingDescriptor_t pool_descriptor;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_descriptor));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_descriptor,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        filter_h, filter_w,
        padding_h, padding_w, // zero-padding
        u, v // stride
    ));


    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));



    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(
        pool_descriptor,
        input_descriptor,
        &output_n,
        &output_c,
        &output_h,
        &output_w));


   

    if (dataformat == 0) {
        assert(output->shape[0] == output_n);
        assert(output->shape[1] == output_c);
        assert(1 == output_h);
        assert(output->shape[2] == output_w);
    }
    else {
        assert(output->shape[0] == output_n);
        assert(output->shape[2] == output_c);
        assert(1 == output_h);
        assert(output->shape[1] == output_w);
    }

    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        output_n,
        output_c,
        output_h,
        output_w));


    auto alpha = 1.0f, beta = 1.0f;
    CUDNN_CALL(cudnnPoolingBackward(handle,
        pool_descriptor,
        &alpha,
        output_descriptor,
        output->data,
        output_descriptor,
        doutput->data,
        input_descriptor,
        input->data,
        &beta,
        input_descriptor,
        dinput->data));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));
    return 0;
}



int DLGpuPooling2DForward(const DLArrayHandle input,
    DLArrayHandle output,
    cudnnTensorFormat_t dataformat,
    const int padding_h,
    const int padding_w,
    const int u,
    const int v,
    const int filter_h,
    const int filter_w)
{

    int input_n = input->shape[0];
    int input_c = input->shape[3];
    int input_h = input->shape[1];
    int input_w = input->shape[2];
    if (dataformat == 0) {
        input_n = input->shape[0];
        input_c = input->shape[1];
        input_h = input->shape[2];
        input_w = input->shape[3];
    }



    int output_n;
    int output_c;
    int output_h;
    int output_w;

    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));


    cudnnPoolingDescriptor_t pool_descriptor;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_descriptor));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_descriptor,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        filter_h, filter_w,
        padding_h, padding_w, // zero-padding
        u, v // stride
    ));
    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));




    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(
        pool_descriptor,
        input_descriptor,
        &output_n,
        &output_c,
        &output_h,
        &output_w));

    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        output_n,
        output_c,
        output_h,
        output_w));


    auto alpha = 1.0f, beta = 1.0f;
    CUDNN_CALL(cudnnPoolingForward(handle,
        pool_descriptor,
        &alpha,
        input_descriptor,
        input->data,
        &beta,
        output_descriptor,
        output->data));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));
   return 0;
}



int DLGpuPooling2DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
    cudnnTensorFormat_t dataformat,
    const int padding_h,
    const int padding_w,
    const int u,
    const int v,
    const int filter_h,
    const int filter_w)
{
    int input_n = input->shape[0];
    int input_c = input->shape[3];
    int input_h = input->shape[1];
    int input_w = input->shape[2];

    if (dataformat == 0) {
        input_n = input->shape[0];
        input_c = input->shape[1];
        input_h = input->shape[2];
        input_w = input->shape[3];
    }



    int output_n;
    int output_c;
    int output_h;
    int output_w;

    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));


    cudnnPoolingDescriptor_t pool_descriptor;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_descriptor));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_descriptor,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        filter_h, filter_w,
        padding_h, padding_w, // zero-padding
        u, v // stride
    ));
    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w));




    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(
        pool_descriptor,
        input_descriptor,
        &output_n,
        &output_c,
        &output_h,
        &output_w));

    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        output_n,
        output_c,
        output_h,
        output_w));


    auto alpha = 1.0f, beta = 1.0f;
    CUDNN_CALL(cudnnPoolingBackward(handle,
        pool_descriptor,
        &alpha,
        output_descriptor,
        output->data,
        output_descriptor,
        doutput->data,
        input_descriptor,
        input->data,
        &beta,
        input_descriptor,
        dinput->data));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));
   return 0;
}




int DLGpuPooling3DForward(const DLArrayHandle input,
    DLArrayHandle output,
    cudnnTensorFormat_t dataformat,
    const int padding1,
    const int padding2,
    const int padding3,
    const int s1,
    const int s2,
    const int s3,
    const int filter1,
    const int filter2,
    const int filter3)
{
    assert(input->ndim == 5);


    int* input_shape, * output_shape, * filter_shape, * inputstrides, * outputstrides;


    int* padA, * filterStrideA;
    input_shape = (int*)malloc(sizeof(int) * 5);
    filter_shape = (int*)malloc(sizeof(int) * 3);
    output_shape = (int*)malloc(sizeof(int) * 5);
    inputstrides = (int*)malloc(sizeof(int) * 5);
    outputstrides = (int*)malloc(sizeof(int) * 5);
    padA = (int*)malloc(sizeof(int) * 3);
    filterStrideA = (int*)malloc(sizeof(int) * 3);
    for(int i=0;i<5;i++)
    {
        input_shape[i]= input->shape[i];
    }
    filter_shape[0] = filter1;
    filter_shape[1] = filter2;
    filter_shape[2] = filter3;
    filterStrideA[0] = s1;
    filterStrideA[1] = s2;
    filterStrideA[2] = s3;
    padA[0] = padding1;
    padA[1] = padding2;
    padA[2] = padding3;


    inputstrides[4] = 1;
    for (int i = 1; i < 5; i++) {
        inputstrides[4 - i] = inputstrides[5 - i] * input_shape[5 - i];
    }

    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));


    cudnnPoolingDescriptor_t pool_descriptor;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_descriptor));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(pool_descriptor,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        3,
        filter_shape,
        padA,
        filterStrideA));
    

    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        input_shape));


    CUDNN_CALL(cudnnGetPoolingNdForwardOutputDim(
        pool_descriptor,
        input_descriptor,
        5,
        output_shape));

    assert(output->shape[0] == output_shape[0]);
    assert(output->shape[1] == output_shape[1]);
    assert(output->shape[2] == output_shape[2]);
    assert(output->shape[3] == output_shape[3]);
    assert(output->shape[4] == output_shape[4]);

    outputstrides[4] = 1;
    for (int i = 1; i < 5; i++) {
        outputstrides[4 - i] = outputstrides[5 - i] * output_shape[5 - i];
    }


    //output
    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        output_shape));


    auto alpha = 1.0f, beta = 1.0f;
    CUDNN_CALL(cudnnPoolingForward(handle,
        pool_descriptor,
        &alpha,
        input_descriptor,
        input->data,
        &beta,
        output_descriptor,
        output->data));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));
    return 0;

}




int DLGpuPooling3DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
    cudnnTensorFormat_t dataformat,
    const int padding1,
    const int padding2,
    const int padding3,
    const int s1,
    const int s2,
    const int s3,
    const int filter1,
    const int filter2,
    const int filter3)
{
    assert(input->ndim == 5);


    int* input_shape, * output_shape, * filter_shape, * inputstrides, * outputstrides;


    int* padA, * filterStrideA;
    input_shape = (int*)malloc(sizeof(int) * 5);
    filter_shape = (int*)malloc(sizeof(int) * 3);
    output_shape = (int*)malloc(sizeof(int) * 5);
    inputstrides = (int*)malloc(sizeof(int) * 5);
    outputstrides = (int*)malloc(sizeof(int) * 5);
    padA = (int*)malloc(sizeof(int) * 3);
    filterStrideA = (int*)malloc(sizeof(int) * 3);
    for(int i=0;i<5;i++)
    {
        input_shape[i]= input->shape[i];
    }
    filter_shape[0] = filter1;
    filter_shape[1] = filter2;
    filter_shape[2] = filter3;
    filterStrideA[0] = s1;
    filterStrideA[1] = s2;
    filterStrideA[2] = s3;
    padA[0] = padding1;
    padA[1] = padding2;
    padA[2] = padding3;


    inputstrides[4] = 1;
    for (int i = 1; i < 5; i++) {
        inputstrides[4 - i] = inputstrides[5 - i] * input_shape[5 - i];
    }

    //handle
    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));


    cudnnPoolingDescriptor_t pool_descriptor;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_descriptor));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(pool_descriptor,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        3,
        filter_shape,
        padA,
        filterStrideA));


    //input
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(input_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        input_shape));


    CUDNN_CALL(cudnnGetPoolingNdForwardOutputDim(
        pool_descriptor,
        input_descriptor,
        5,
        output_shape));

    assert(output->shape[0] == output_shape[0]);
    assert(output->shape[1] == output_shape[1]);
    assert(output->shape[2] == output_shape[2]);
    assert(output->shape[3] == output_shape[3]);
    assert(output->shape[4] == output_shape[4]);

    outputstrides[4] = 1;
    for (int i = 1; i < 5; i++) {
        outputstrides[4 - i] = outputstrides[5 - i] * output_shape[5 - i];
    }


    //output
    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensorNdDescriptorEx(output_descriptor,
        dataformat,
        CUDNN_DATA_FLOAT,
        5,
        output_shape));


    auto alpha = 1.0f, beta = 1.0f;
    CUDNN_CALL(cudnnPoolingBackward(handle,
        pool_descriptor,
        &alpha,
        output_descriptor,
        output->data,
        output_descriptor,
        doutput->data,
        input_descriptor,
        input->data,
        &beta,
        input_descriptor,
        dinput->data));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_descriptor));
    CUDNN_CALL(cudnnDestroy(handle));
    return 0;

}







