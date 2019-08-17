"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf cumprod
"""


from te import tvm
from te import platform as cce
from topi.cce import util
from te.platform.cce_build import build_config
from functools import reduce as functools_reduce

SHAPE_SIZE_LIMIT =  100000000  # shape limit for tf_cumprod

def build_for(input_data, output_data ,ib, shape, cumprod_axis, idx, idx_list, exclusive, reverse):
    """
    build for loop recursion.

    :param input_data: input_buffer
    :param output_data: output_buffer
    :param ib: ir builder
    :param shape: input shape, the recursion layer size equals len(shape), and loop is (0, shape[xx])
    :param cumprod_axis: cumprod axis
    :param idx: idx is the loop layer, the loop inner index named like "i0, i1, ..."
    :param idx_list:  idx list store the every loop inner index, idx_list store ( (i0, shpae[0]), (i1, shape[1]), ...  )
    :param exclusive: exclusive prod self
    :param reverse: reverse prod direction
    :return: none
    """
    loop_cnt = shape[0]
    # cut current after handle this dim
    shape = shape[1:] if len(shape)>1 else []
    # build for loop
    with ib.for_range(0, loop_cnt, for_type = "serial", name = "i"+str(idx)) as inner_idx:
        # store loop index and loop count
        idx_list.append([inner_idx, loop_cnt])
        # if reset shape > 0 , should build for by recursion
        if len(shape) > 0:
            idx = idx + 1
            build_for(input_data, output_data, ib, shape,cumprod_axis, idx, idx_list, exclusive, reverse)
        else:
            # for example the calculate is like  out[ i2 + 5 * (i1 + 4*i0) ] = out[ i2 + 5 * ( (i1-1) + 4*i0) ] * input[ i2 + 5 * (i1 + 4 * i0) ]
            # build cur_offset and pre_offset, then the calculate will like out[cur_offset] = out[pre_offset] * input[cur_offset]
            # if exclusive = True the calcuate will like out[cur_offset] = out[pre_offset] * input[pre_offset]
            if reverse:
                idx_list[cumprod_axis] =  idx_list[cumprod_axis][:]
                # if reverse = True , the before example is like :out[ i2 + 5 * ( (4 - i1) + 4*i0) ] = out[ i2 + 5 * ( (4 - i1 + 1) + 4*i0) ] * input[ i2 + 5 * ( (4 - i1) + 4 * i0) ]
                # from this we can know the idx is like loop_cnt - 1 - i1, i1 is idx_list[cumprod_axis][0]
                idx_list[cumprod_axis][0] = idx_list[cumprod_axis][1] - 1 - idx_list[cumprod_axis][0]
                pre_idx_list = idx_list[:]
                pre_idx_list[cumprod_axis] = pre_idx_list[cumprod_axis][:]
                # if reverse = True, the pre idx should be add 1
                pre_idx_list[cumprod_axis][0] = pre_idx_list[cumprod_axis][0] + 1
            else:
                pre_idx_list = idx_list[:]
                pre_idx_list[cumprod_axis] = pre_idx_list[cumprod_axis][:]
                # if reverse = False, the pre idx should minus 1
                pre_idx_list[cumprod_axis][0] = pre_idx_list[cumprod_axis][0] - 1

            # the offset is like i3 + i3_loop_cnt * ( i2 + i2_loop_cnt * (i1 + i1_loop_cnt* (...)) )
            # the idx_list is like :[ [i0, i0_loop_cnt ] , [i1, i1_loop_cnt ]...], use reduce to calculate offset
            cur_offset_tmp = functools_reduce(lambda i,j: [i[0] * j[1] + j[0], 0 ], idx_list) if len(idx_list)>0 else [0,0]
            cur_offset = cur_offset_tmp[0]
            pre_offset_tmp = functools_reduce(lambda i, j: [i[0] * j[1] + j[0], 0], pre_idx_list) if len(pre_idx_list) > 0 else [0, 0]
            pre_offset = pre_offset_tmp[0]

            if reverse:
                with ib.if_scope(idx_list[cumprod_axis][0] == idx_list[cumprod_axis][1] - 1):
                    # Calculate the first output value:
                    # if  reverse = True  exclusive =True :the before example is like : out[ i2 + 5 * ( (4 - i1) + 4*i0) ] = 1 ,when i1 == (4 - 1).
                    # if  reverse = True  exclusive =False :the before example is like : out[cur_offset] = input[cur_offset] ,when i1 == (4 - 1).
                    if exclusive:
                        output_data[cur_offset] = tvm.const(1, input_data.dtype)
                    else:
                        output_data[cur_offset] = input_data[cur_offset]
                with ib.else_scope():
                    #Calculate the output value in other cases: 
                    #The output value is always equal to the previous output value  multiplied by the current input value.
                    if exclusive:
                        output_data[cur_offset] = output_data[pre_offset ] * input_data[pre_offset]
                    else:
                        output_data[cur_offset] = output_data[pre_offset] * input_data[cur_offset]
            else:
                with ib.if_scope(idx_list[cumprod_axis][0] == 0):
                    # Calculate the first output value:
                    # if  reverse = False  exclusive =True :the before example is like : out[ i2 + 5 * ( (4 - i1) + 4*i0) ] = 1 ,when i1 = 0.
                    # if  reverse = False  exclusive =False :the before example is like : out[cur_offset] = input[cur_offset] ,when i1 = 0.
                    if exclusive:
                        output_data[cur_offset] = tvm.const(1, input_data.dtype)
                    else:
                        output_data[cur_offset] = input_data[cur_offset]
                with ib.else_scope():
                    #Calculate the output value in other cases:
                    #The output value is always equal to the previous output value  multiplied by the current input value.
                    if exclusive:
                        output_data[cur_offset] = output_data[pre_offset] * input_data[pre_offset]
                    else:
                        output_data[cur_offset] = output_data[pre_offset] * input_data[cur_offset]


def cumprod_ir(input, output, axis = 0, exclusive=False, reverse=False, dim_size_one=False):
    """
    build cumprod ir
    example 1: input.shape = [3,4,5]; axis = 1; exclusive = False, reverse = False; dim_size_one = false, ir like:
        for i0 in (0, 3)
            for i1 in (0, 4)
                for i2 in (0, 5)
                    if i1 == 0:
                        out[ i2 + 5 * (i1 + 4*i0) ] = input[ i2 + 5 * (i1 + 4 * i0) ]
                    else:
                        out[ i2 + 5 * (i1 + 4*i0) ] = out[ i2 + 5 * ( (i1-1) + 4*i0) ] * input[ i2 + 5 * (i1 + 4 * i0) ]

    example 2 : input.shape = [3,4,5]; axis = 1; exclusive = True, reverse = False; dim_size_one = false, ir like:
        for i0 in (0, 3)
            for i1 in (0, 4)
                for i2 in (0, 5)
                    if i1 == 0:
                        out[ i2 + 5 * (i1 + 4*i0) ] = 1
                    else:
                        out[ i2 + 5 * (i1 + 4*i0) ] = out[ i2 + 5 * ( (i1-1) + 4*i0) ] * input[ i2 + 5 * ( (i1-1) + 4 * i0) ]

    example 3 : input.shape = [3,4,5]; axis = 1; exclusive = True, reverse = True; dim_size_one = false, ir like:
        for i0 in (0, 3)
            for i1 in (0, 4)
                for i2 in (0, 5)
                    if i1 == (4 - 1):
                        out[ i2 + 5 * ( (4 - i1) + 4*i0) ] = 1
                    else:
                        out[ i2 + 5 * ( (4 - i1) + 4*i0) ] = out[ i2 + 5 * ( (4 - i1 + 1) + 4*i0) ] * input[ i2 + 5 * ( (4 - i1 + 1) + 4 * i0) ]

    example 4 : input.shape = [3,4,5]; axis = 1; exclusive = False, reverse = True; dim_size_one = false, ir like:
        for i0 in (0, 3)
            for i1 in (0, 4)
                for i2 in (0, 5)
                    if i1 == (4 - 1):
                        out[ i2 + 5 * ( (4 - i1) + 4*i0) ] = input[ i2 + 5 * ( (4 - i1) + 4*i0) ]
                    else:
                        out[ i2 + 5 * ( (4 - i1) + 4*i0) ] = out[ i2 + 5 * ( (4 - i1 + 1) + 4*i0) ] * input[ i2 + 5 * ( (4 - i1) + 4 * i0) ]

    :param input: the input tensor
    :param output: the output tensor
    :param axis: axis cumprod at axis
    :param exclusive: True or false
    :param reverse: reverse cumprod direct
    :param dim_size_one: the input.shape[axis] == 1 and exclusive = True
    :return: out_tensor
    """
    ib = tvm.ir_builder.create()
    p_in = ib.buffer_ptr(input)
    shape = input.shape[:]

    # shape[axis]==1 and exclusive = True, build ir will like :
    # for i0 in (0, m):
    #   for i1 in (0, 1):
    #       out[ i1 + m ] = 1
    # here the input was not used, this will call problem, we must use input.
    # we can do  input * (1) - input + tmp_out
    if dim_size_one:
        tmp_out = ib.allocate(output.dtype, shape, name="tmp_out", scope=cce.scope_aicpu)
        # build for loop, this will be a recursion
        build_for(p_in, tmp_out, ib, shape, axis, 0, [], exclusive, reverse)

        tmp_var = ib.allocate(output.dtype, shape, name="tmp_var", scope=cce.scope_aicpu)
        loop_cnt = functools_reduce(lambda i, j: i * j, shape)
        with ib.for_range(0, loop_cnt, for_type="serial", name="i") as inner_i:
            tmp_var[inner_i] = tvm.const(1, input.dtype)

        with ib.for_range(0, loop_cnt, for_type="serial", name="j") as inner_j:
            tmp_var[inner_j] = tmp_var[inner_j] * p_in[inner_j]

        with ib.for_range(0, loop_cnt, for_type="serial", name="l") as inner_l:
            tmp_var[inner_l] = tmp_var[inner_l] - p_in[inner_l]

        p_out = ib.buffer_ptr(output)
        with ib.for_range(0, loop_cnt, for_type="serial", name="k") as inner_k:
            p_out[inner_k] = tmp_out[inner_k] + tmp_var[inner_k]
    else:
        p_out = ib.buffer_ptr(output)
        # build for loop, this will be a recursion
        build_for(p_in, p_out, ib, shape, axis, 0, [], exclusive, reverse)
    return ib.get()

@util.check_input_type((list, tuple), str, int, bool, bool, str, bool, bool)
def custom_cumprod(shape, dtype, axis=0, exclusive=False, reverse=False, kernel_name="cce_tf_cumprod", need_build=False, need_print=False):
    """
    algorithm: tf_cumprod

    compute the cumulative prod of the tensor x along axis.

    Parameters
    ----------
    shape : shape of data

    dtype : the data type, assume src_dtype equals dst_dtype, only support float16, float32, int32

    axis: A Tensor of type int32 (default: 0). Must be in the range [-rank(x), rank(x)).

    exclusive: If True, perform exclusive cumprod.

    reverse: A bool (default: False).

    kernel_name : cce kernel name, default value is "cce_tf_cumprod"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    A Tensor. Has the same type as x.

    """
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    check_list = ["float16", "float32", "int32"]
    if not (dtype.lower() in check_list):
        raise RuntimeError("abs_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    inp_dtype = dtype.lower()
    shape_len = len(shape)
    axis = util.axis_check(shape_len, axis)

    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)
    res = tvm.extern([shape], [data_input],
                                    lambda ins,outs:cumprod_ir(ins[0], output=outs[0],axis= axis, exclusive=exclusive, reverse=reverse, dim_size_one=(exclusive and shape[axis] == 1)),
                                 dtype=[inp_dtype], name="cumprod")
    s = tvm.create_schedule([res.op])
    if need_print:
        with build_config:
            print(tvm.lower(s, [data_input, res], simple_mode=True))
    if need_build:
        with build_config:
            tvm.build(s, [data_input, res], "cce", name=kernel_name)
