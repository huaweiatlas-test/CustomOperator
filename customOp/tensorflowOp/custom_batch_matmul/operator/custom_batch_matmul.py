"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from te import tvm
from topi.cce import util
from te.platform.cce_build import build_config

SHAPE_SIZE_LIMIT =  100000000  # shape limit for tf_batch_matmul
SHAPE_SIZE_FP16_LIMIT = 6500000000
SHAPE_SIZE_FP32_LIMIT = 5000000000
SHAPE_SIZE_INT32_LIMIT = 9700000000

@util.check_input_type((list, tuple), (list, tuple), str, bool, bool, str, bool, bool)
def custom_batch_matmul(shape_x, shape_y, dtype, trans_a=False, trans_b=False, kernel_name = "cce_tf_batch_matmul", need_build = False, need_print = False):

    """
    Multiplies slices of two tensors in batches(each slice can be viewed
    as an element of a batch), the output is of the same batch size.

    Each of the individual slices can optionally be transposed before
    multiplication by setting the trans_a or trans_b flag to True, which
    are by default False. The input tensors are 2-D or higher with the
    shape [..., r_x, c_x] and [..., r_y, c_y]. The output tensor is 2-D
    or higher with the shape [..., r_o, c_o], where
    r_o = c_x if trans_a else r_x
    c_o = r_y if trans_b else c_y

    Parameters
    ----------
    shape_x : shape of the first tensor x with rank > 1

    shape_y : shape of the second tensor y with the same type and shape with x

    dtype : the data type, support int8, uint8,float16,float32,int32

    kernel_name : cce kernel name, default value is "cce_batch_matmul"

    trans_a : if True, shape_x is transposed before multiplication

    trans_b : if True, shape_y is transposed before multiplication

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)

    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)

    data_dtype = dtype.lower()
    check_list = ["int8", "uint8", "float16", "float32", "int32"]
    if not (data_dtype in check_list):
        raise RuntimeError("batch_matmul_cce ony supports %s while dtype is %s" % (",".join(check_list), dtype))

    def transpose_tensor(shape, size):
        """Transpose the shape, e.g., the shape [..., r_x, c_x] is transposed to [..., c_x, r_x].

        Parameters
        ----------
        shape : shape of a tensor

        size : length of the shape

        Returns
        -------
        shape_ori : the transposed shape
        """
        shape_ori = ()
        if size == 1:
            shape_ori = shape_ori + shape
        elif size == 2:
            shape_ori = shape_ori + (shape[1], ) + (shape[0], )
        else:
            shape_ori = shape_ori + (shape[:(size-2)]) + (shape[size-1], ) + (shape[size-2], )
        return shape_ori

    def check_matmul(shape_x, shape_y):
        """Check whether batch_matmul is supported or not.

        Parameters
        ----------
        shape_x : shape of the first tensor x

        shape_y : shape of the second tensor y with the same type and shape with x

        Returns
        -------
        None
        """
        len_x = len(shape_x)
        len_y = len(shape_y)
        if (len_x < 2) or (len_y < 2):
            raise RuntimeError("Only tensors of rank>=2 are supported!")
        if (shape_x[len_x-1] != shape_y[len_y-2]):
            raise RuntimeError("Invalid matrix multiplication for the inner 2 dimensions!")
        if (len_x == len_y) and (len_x > 2):
            for i in range(len_x - 2):
                if (shape_x[i] != shape_y[i]):
                    raise RuntimeError("Outer dimensions do not match!")
            return
        elif (len_x == len_y) and (len_x == 2):
            return
        else:
            raise RuntimeError("The input tensors are not with the same rank!")

    def _compute(output_shape, x, y, K, trans_a, trans_b, *indices):
        """matmul compuation in terms of the output shape and the transposes

        Parameters
        ----------
        output_shape : the final output shape, e.g., shape_x = (2, 6),
            shape_y = (8, 2), trans_a = True, True_b = True, then, output_shape = (6, 8).

        x : the first input tensor according to shape_x.

        y : the second input tensor according to shape_y.

        K : the number of the axis for sum, in the above example, K = 2.

        trans_a : if True, x needs to be transposed.

        trans_b : if True, y needs to be transposed.

        *indices : the output shape space for tvm.compute.

        Returns
        -------
        tvm.Tensor
        """
        N = len(output_shape)
        k = tvm.reduce_axis((0, K), 'k')
        if trans_a == True and trans_b == False:
            # For example, A: (6, 7, 8), B: (6, 7, 9), so the length is N = 3
            # C = A' * B : (6, 8, 9), A' means the transpose of A
            # indices means the space of (6, 8, 9), k = 7
            # x_indices = indices[:1]+(7, )+indices[1:2] = (6, 7, 8)
            # y_indices = indices[:1]+(7, )+indices[2:] = (6, 7, 9)
            x_indices = indices[:(N-2)]+(k, )+indices[(N-2):(N-1)]
            y_indices = indices[:(N-2)]+(k, )+indices[(N-1):]
            return tvm.sum(x(*x_indices)*y(*y_indices), axis=k)
        elif trans_a == False and trans_b == True:
            # For example, A: (6, 7, 8), B: (6, 9, 8), C = A * B' : (6, 7, 9)
            # indices means the space of (6, 7, 9), N=3, k = 8
            # x_indices = indices[:2]+(8, ) = (6, 7, 8)
            # y_indices = indices[:1]+indices[2:]+(8, ) = (6, 9, 8)
            x_indices = indices[:(N-1)]+(k, )
            y_indices = indices[:(N-2)]+indices[(N-1):]+(k, )
            return tvm.sum(x(*x_indices)*y(*y_indices), axis=k)
        elif trans_a == True and trans_b == True:
            # For example, A: (6, 8, 10), B: (6, 12, 8), C = A' * B' : (6, 10, 12)
            # indices means the space of (6, 10, 12), N=3, k = 8
            # x_indices = indices[:1]+(8, )+indices[1:2] = (6, 8, 10)
            # y_indices = indices[:1]+indices[2:]+(8, ) = (6, 12, 8)
            x_indices = indices[:(N-2)]+(k, )+indices[(N-2):(N-1)]
            y_indices = indices[:(N-2)]+indices[(N-1):]+(k, )
            return tvm.sum(x(*x_indices)*y(*y_indices), axis=k)
        else:
            # For example, A: (6, 15, 16), B: (6, 16, 18), C = A * B : (6, 15, 18)
            # indices means the space of (6, 15, 18), N=3, k = 16
            # x_indices = indices[:2]+(16, ) = (6, 15, 16)
            # y_indices = indices[:1]+(16, )+indices[2:] = (6, 16, 18)
            x_indices = indices[:(N-1)]+(k, )
            y_indices = indices[:(N-2)]+(k, )+indices[(N-1):]
            return tvm.sum(x(*x_indices)*y(*y_indices), axis=k)

    def check_supportted_shape_size(shape_x, shape_y, limit, trans_a, trans_b):
        """
        check shape size for operator
        ----------
        shape: shape of data

        limit: limit of the product

        Returns
        -------
        None
        """
        # This function is used to check whether the shape is too large to cause a timeout.
        # shape_x = (a,b,c,d,e,k)  shape_y = (a,b,c,d,k,f)
        # t_1 : time consumed by each addition operation
        # t_2 : time consumed by each multiplication operation
        # t_all : time consumed by a complete calculation
        # t_all is approximately equal to (a * b * c * d) * (e * k * f) * (t_1 + t_2)
        # As (t_1 + t_2) is a constant, so t_all is proportional to (a * b * c * d * e * k * f)

        len_x = len(shape_x)
        len_y = len(shape_y)
        if (len_x < 2) or (len_y < 2):
            raise RuntimeError("Only tensors of rank>=2 are supported!")

        shape_x = list(shape_x)
        shape_y = list(shape_y)

        tmp_shape_x = shape_x[:]
        if trans_a:
            tmp_shape_x = shape_x[:-2] + [shape_x[-1], shape_x[-2]]

        tmp_shape_y = shape_y[:]
        if trans_b:
            tmp_shape_y = shape_y[:-2] + [shape_y[-1], shape_y[-2]]

        union_shape = tmp_shape_x + [tmp_shape_y[-1]]

        union_size = reduce(lambda i,j: i*j, union_shape)

        if union_size > limit:
            raise RuntimeError("the shape is too large to calculate" )

    if data_dtype in ["float16", "float32", "int32"]:
        type_shape_map = {
            'float16':SHAPE_SIZE_FP16_LIMIT,
            'float32':SHAPE_SIZE_FP32_LIMIT,
            'int32':SHAPE_SIZE_INT32_LIMIT
        }

        check_supportted_shape_size(shape_x, shape_y, type_shape_map[data_dtype], trans_a, trans_b)

    x_size = len(shape_x)
    y_size = len(shape_y)
    shape_a = shape_x
    shape_b = shape_y
    if trans_a == True:
        shape_x = transpose_tensor(shape_x, x_size)

    if trans_b == True:
        shape_y = transpose_tensor(shape_y, y_size)

    check_matmul(shape_x, shape_y)
    last_axis = shape_x[x_size - 1]

    x = tvm.placeholder(shape_a, name="input_1", dtype=data_dtype)
    y = tvm.placeholder(shape_b, name="input_2", dtype=data_dtype)

    # output shape
    output_shape = ()
    for i in range(x_size - 1):
        output_shape = output_shape + (shape_x[i], )
    output_shape = output_shape + (shape_y[x_size - 1], )
    result = tvm.compute(output_shape, lambda *indices: _compute(output_shape, x, y, last_axis, trans_a, trans_b, *indices), name="result")
    s = tvm.create_schedule(result.op)

    if need_print:
        with build_config:
            print( tvm.lower(s, [x, y, result], simple_mode=True))
    if need_build:
        with build_config:
            tvm.build(s, [x, y, result], "cce", name=kernel_name)

