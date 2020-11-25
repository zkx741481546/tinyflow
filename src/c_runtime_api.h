/*!
 * \file c_runtime_api.h
 * \brief DL runtime library.
 *
 */

#ifndef TINYFLOW_RUNTIME_C_RUNTIME_API_H_
#define TINYFLOW_RUNTIME_C_RUNTIME_API_H_

#ifdef __cplusplus
#define TINYFLOW_EXTERN_C extern "C"
#else
#define TINYFLOW_EXTERN_C
#endif

#include "dlarray.h"
#include <stddef.h>
#include <stdint.h>
#include <cudnn.h>
TINYFLOW_EXTERN_C {
  /*! \brief type of array index. */
  typedef int64_t index_t;

  /*! \brief the array handle */
  typedef DLArray *DLArrayHandle;
  /*!
   * \brief The stream that is specific to device
   * can be NULL, which indicates the default one.
   */
  typedef void *DLStreamHandle;

  typedef enum {
      SAME = 0,
      VALID = 1,
  } paddingStatus_t;

//  typedef enum {
//      CUDNN_TENSOR_NCHW = 0,
//      CUDNN_TENSOR_NHWC = 1,
//  } cudnnTensorFormat_t

  // Array related apis for quick proptying
  /*!
   * \brief Allocate a nd-array's memory,
   *  including space of shape, of given spec.
   *
   * \param shape The shape of the array, the data content will be copied to out
   * \param ndim The number of dimension of the array.
   * \param ctx The ctx this array sits on.
   * \param out The output handle.
   * \return 0 when success, -1 when failure happens
   */
  int DLArrayAlloc(const index_t *shape, index_t ndim, DLContext ctx,
                   DLArrayHandle *out);

  /*!
   * \brief Free the DL Array.
   * \param handle The array handle to be freed.
   * \return 0 when success, -1 when failure happens
   */
  int DLArrayFree(DLArrayHandle handle);

  /*!
   * \brief Copy the array, both from and to must be valid during the copy.
   * \param from The array to be copied from.
   * \param to The target space.
   * \param stream The stream where the copy happens, can be NULL.
   * \return 0 when success, -1 when failure happens
   */
  int DLArrayCopyFromTo(DLArrayHandle from, DLArrayHandle to,
                        DLStreamHandle stream);

  /*!
   * \brief Set all array elements to given value.
   * \param arr The array to be Set.
   * \param value The target value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuArraySet(DLArrayHandle arr, float value);

  /*!
   * \brief Broadcast input array to output array.
   * \param input The input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Reduce sum input array by axis=0 and store to output.
   * \param input The input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Elementwise add two matrices and store to output.
   * \param matA The left input array.
   * \param matB The right input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                                const DLArrayHandle matB, DLArrayHandle output);

  /*!
   * \brief Add matrix by const and store to output.
   * \param input The input array.
   * \param val The constant.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                       DLArrayHandle output);

  /*!
   * \brief Elementwise multiply two matrices and store to output.
   * \param matA The left input array.
   * \param matB The right input array.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixElementwiseMultiply(
      const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output);

  /*!
   * \brief Multiply matrix by const and store to output.
   * \param input The input array.
   * \param val The constant.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                                 DLArrayHandle output);

  /*!
   * \brief Matrix multiply two matrices and store to output.
   * \param matA The left input array.
   * \param transposeA Whether matA needs to be transposed
   * \param matB The right input array.
   * \param transposeB Whether matB needs to be transposed
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                          const DLArrayHandle matB, bool transposeB,
                          DLArrayHandle matC);

  /*!
   * \brief Compute relu on all array elements, and store to output.
   * \param input The input array.
   * \param output The output value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Compute relu gradient, and store to output.
   * \param input The input array.
   * \param in_grad The input gradients value.
   * \param output The output array.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                        DLArrayHandle output);

  /*!
   * \brief Compute softmax on matrix, and store to output.
   * \param input The input array.
   * \param output The output value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output);

  /*!
   * \brief Compute softmax_cross_entropy.
   *  np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
   * \param input_a The y array.
   * \param input_b The y_ array.
   * \param output The output value.
   * \return 0 when success, -1 when failure happens
   */
  int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output);




  int DLGpuMatrixExp(const DLArrayHandle input, DLArrayHandle output);


  int DLGpuMatrixLog(const DLArrayHandle input, DLArrayHandle output);


  int DLGpuMatrixReverse(const DLArrayHandle input, DLArrayHandle output);

  int DLGpuMatrixPow(const DLArrayHandle input,const float val, DLArrayHandle output);



  int DLGpuReduceSumAxisN(const DLArrayHandle input, DLArrayHandle output, const int axis);


  int DLGpuReduceSumAll(const DLArrayHandle input, DLArrayHandle output);

  int DLGpuReduceSumAllBackward(const DLArrayHandle input, DLArrayHandle output);

  int DLGpuReduceSumAxisNBackward(const DLArrayHandle input, DLArrayHandle output, const int axis);

//juanji
  int DLGpuConvolution1DForward(const DLArrayHandle input,
      const DLArrayHandle filter,
      DLArrayHandle output,
      cudnnTensorFormat_t dataformat,
      const paddingStatus_t padding,
      const int v          /*filter stride */);






  int DLGpuConvolution2DForward(const DLArrayHandle input,
      const DLArrayHandle filter,
      DLArrayHandle output,
      cudnnTensorFormat_t dataformat,
      const paddingStatus_t padding,
      const int u,/* vertical filter stride */
      const int v/* horizontal filter stride */);


  int DLGpuConvolution3DForward(const DLArrayHandle input,
      const DLArrayHandle filter,
      DLArrayHandle output,
      cudnnTensorFormat_t dataformat,
      const paddingStatus_t padding,
      const int s1,
      const int s2,
      const int s3);

   int DLGpuConvolution1DBackward(const DLArrayHandle input,
      const DLArrayHandle doutput,
      const DLArrayHandle filter,
      DLArrayHandle dfilter,
      DLArrayHandle dinput,
      cudnnTensorFormat_t dataformat,
      const paddingStatus_t padding,
      const int v          /*filter stride */);



  int DLGpuConvolution2DBackward(const DLArrayHandle input,
      const DLArrayHandle doutput,
      const DLArrayHandle filter,
      DLArrayHandle dfilter,
      DLArrayHandle dinput,
      cudnnTensorFormat_t dataformat,
      const paddingStatus_t padding,
      const int u,/* vertical filter stride */
      const int v/* horizontal filter stride */);


  int DLGpuConvolution3DBackward(const DLArrayHandle input,
      const DLArrayHandle doutput,
      const DLArrayHandle filter,
      DLArrayHandle dfilter,
      DLArrayHandle dinput,
      cudnnTensorFormat_t dataformat,
      const paddingStatus_t padding,
      const int s1,
      const int s2,
      const int s3);


  int DLGpuPooling1DForward(const DLArrayHandle input,
      DLArrayHandle output,
      cudnnTensorFormat_t dataformat,
      cudnnPoolingMode_t poolingMode,
      const int padding_w,
      const int v,
      const int filter_w);

  int DLGpuPooling2DForward(const DLArrayHandle input,
      DLArrayHandle output,
      cudnnTensorFormat_t dataformat,
      cudnnPoolingMode_t poolingMode,
      const int pad_h,      /* zero-padding height */
      const int pad_w,      /* zero-padding width */
      const int u,          /* vertical filter stride */
      const int v,        /* horizontal filter stride */
      const int filter_h,
      const int filter_w);


  int DLGpuPooling3DForward(const DLArrayHandle input,
      DLArrayHandle output,
      cudnnTensorFormat_t dataformat,
      cudnnPoolingMode_t poolingMode,
      const int padding1,
      const int padding2,
      const int padding3,
      const int s1,
      const int s2,
      const int s3,
      const int filter1,
      const int filter2,
      const int filter3);



  int DLGpuPooling1DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
      cudnnTensorFormat_t dataformat,
      cudnnPoolingMode_t poolingMode,
      const int padding_w,
      const int v,
      const int filter_w);

  int DLGpuPooling2DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
      cudnnTensorFormat_t dataformat,
      cudnnPoolingMode_t poolingMode,
      const int pad_h,      /* zero-padding height */
      const int pad_w,      /* zero-padding width */
      const int u,          /* vertical filter stride */
      const int v,        /* horizontal filter stride */
      const int filter_h,
      const int filter_w);


  int DLGpuPooling3DBackward(const DLArrayHandle input,
    const DLArrayHandle output,
    const DLArrayHandle doutput,
    DLArrayHandle dinput,
      cudnnTensorFormat_t dataformat,
      cudnnPoolingMode_t poolingMode,
      const int padding1,
      const int padding2,
      const int padding3,
      const int s1,
      const int s2,
      const int s3,
      const int filter1,
      const int filter2,
      const int filter3);


  int DLGpuConvolution1DForwardGetOutShape(const int* input_shapes,
      const int* filter_shapes,
      int* output_shapes,
      cudnnTensorFormat_t dataformat,
      const paddingStatus_t padding,
      const int v          /*filter stride */);

  int DLGpuActivationForward(const DLArrayHandle input,
      DLArrayHandle output,
      cudnnTensorFormat_t dataformat,
      cudnnActivationMode_t activationMode);

  int DLGpuActivationBackward(const DLArrayHandle input,
      DLArrayHandle dinput,
      const DLArrayHandle output,
      const DLArrayHandle doutput,
      cudnnTensorFormat_t dataformat,
      cudnnActivationMode_t activationMode);

  int DLGpuPooling1DForwardGetOutShape(const int* input_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    cudnnPoolingMode_t poolingMode,
    const int padding_w,
    const int v,
    const int filter_w);

  int DLGpuConvolution2DForwardGetOutShape(const int* input_shapes,
      const int* filter_shapes,
      int* output_shapes,
      cudnnTensorFormat_t dataformat,
      const paddingStatus_t padding,
      const int u,          /* vertical filter stride */
      const int v          /* horizontal filter stride */);

  int DLGpuConvolution3DForwardGetOutShape(const int* input_shapes,
    const int* filter_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    const paddingStatus_t padding,
    const int s1,
    const int s2,
    const int s3);

  int DLGpuPooling2DForwardGetOutShape(const int* input_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    cudnnPoolingMode_t poolingMode,
    const int padding_h,
    const int padding_w,
    const int u,
    const int v,
    const int filter_h,
    const int filter_w);

  int DLGpuPooling3DForwardGetOutShape(const int* input_shapes,
    int* output_shapes,
    cudnnTensorFormat_t dataformat,
    cudnnPoolingMode_t poolingMode,
    const int padding1,
    const int padding2,
    const int padding3,
    const int s1,
    const int s2,
    const int s3,
    const int filter1,
    const int filter2,
    const int filter3);


  int DLGpuDropoutForward(const DLArrayHandle input,
    DLArrayHandle output,
    cudnnTensorFormat_t dataformat,
    const float dropout,
    const int seed,
    void **reserveSpace_p/*back use*/);

  int DLGpuDropoutBackward(const DLArrayHandle doutput,
    DLArrayHandle dinput,
    cudnnTensorFormat_t dataformat,
    const float dropout,
    const int seed,
    void **reserveSpace_p/*back use*/);


  int DLGpuCrossEntropy(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output);
  int DLGpuL1loss(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output);
  int DLGpuL2loss(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output);
  int DLGpuL1LossGradient(const DLArrayHandle input,
                               const DLArrayHandle input1,
                               const DLArrayHandle grad,
                               DLArrayHandle output);
  int DLGpuL2LossGradient(const DLArrayHandle input,
                               const DLArrayHandle input1,
                               const DLArrayHandle grad,
                               DLArrayHandle output);
  int DLGpuCrossEntropy(const DLArrayHandle input_a,
                               const DLArrayHandle input_b,
                               DLArrayHandle output);
  int DLGpuL1regular(const DLArrayHandle input_a,
                               DLArrayHandle output);
  int DLGpuL2regular(const DLArrayHandle input_a,
                               DLArrayHandle output);
  int DLGpuL1regularGradient(const DLArrayHandle input,
                               const DLArrayHandle grad,
                               DLArrayHandle output);
  int DLGpuL2regularGradient(const DLArrayHandle input,
                               const DLArrayHandle grad,
                               DLArrayHandle output);

} // TINYFLOW_EXTERN_C

#endif // TINYFLOW_RUNTIME_C_RUNTIME_API_H_
