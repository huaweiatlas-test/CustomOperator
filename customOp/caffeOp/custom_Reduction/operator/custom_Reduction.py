"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

caffe reduction layer
"""

from functools import reduce as functools_reduce

from te import tvm
from topi import generic

import te.lang.cce
from topi.cce import util
from te.platform.cce_build import build_config
from te.platform.fusion_manager import fusion_manager

SHAPE_SIZE_LIMIT = 100000000  # shape limit for caffe_reductionLayer


# Since the shape of placeholder created by caffe_reduce is not same as
# input_shape, fusion_op could not process the fusion of two op
# which have different shape. So, caffe_reduce op could not be
# fused until tvm supports reshape in D.
def caffe_reduction_layer_compute(placeholders, shape, dtype, axis, op, coeff,
                                  kernel_name="cce_reductionLayer",
                                  need_build=False, need_print=False):
    data = placeholders[0]
    inp_dtype = dtype.lower()

    axis = util.axis_check(len(shape), axis)
    shape = list(shape)
    shape1 = shape[:axis] + [functools_reduce(lambda x, y: x*y, shape[axis:])]
    shape1, axis = util.shape_refine(shape1, axis)
    if len(axis) == 0:
        axis = [0]
        shape1 = [1] + shape1

    if op == "ASUM":
        data_tmp_input = te.lang.cce.vabs(data)
        cof = coeff
        tmp = te.lang.cce.vmuls(data_tmp_input, cof)
    elif op == "SUMSQ":
        data_tmp_input = te.lang.cce.vmul(data, data)
        cof = coeff
        tmp = te.lang.cce.vmuls(data_tmp_input, cof)
    elif op == "MEAN":
        size = shape1[-1]
        cof = float(coeff)*(size**(-1))
        if inp_dtype == "int8" \
            or inp_dtype == "uint8":
            data1 = te.lang.cce.vmuls(data, 1.0)
            data_cast = te.lang.cce.cast_to(data1, "float32")
            tmp = te.lang.cce.vmuls(data_cast, cof)
        else:
            tmp = te.lang.cce.vmuls(data, cof)
    elif op == "SUM":
        cof = coeff
        data_tmp_input = te.lang.cce.vmuls(data, cof)
        tmp = data_tmp_input

    res = te.lang.cce.sum(tmp, axis=axis)
    # Although the data type (int8/uint8) has changed,
    # the data values remain integer
    # during the calculation of other operators (SUM/ASUM/SUMSQ).
    if op != "MEAN":
        res = te.lang.cce.cast_to(res, inp_dtype, f1628IntegerFlag=True)
    return res


@util.check_input_type((list, tuple), str, int, str, (int, float), str, bool,
                       bool)
def custom_Reduction(shape, dtype, axis, op, coeff,
                              kernel_name="cce_reductionLayer",
                              need_build=False, need_print=False):
    """
    Reduce a tensor on a certain axis, and scale output with coeff

    Parameters
    ----------
    shape : shape of data

    dtype : source data type, only support float16, float32, int8, uint8

    axis : the first axis to reduce, may be negative to index from the end
           (e.g., -1 for the last axis).
           If axis == 0, the output Blob always has the empty shape (count 1),
           performing reduction across the entire input.

    op : can only be one of "SUM, ASUM (sum of abs), SUMSQ (sum of sqr), MEAN"

    coeff : scale for output

    kernel_name : cce kernel name, default value is "cce_reductionLayer"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None

    """
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)

    check_list = ["float16", "float32", "int8", "uint8"]
    if not (dtype.lower() in check_list):
        raise RuntimeError(
            "reductionLayer_cce only support %s while dtype is %s"
            % (",".join(check_list), dtype))

    ReductionOp = ("SUM", "ASUM", "SUMSQ", "MEAN")

    if type(axis) != int:
        raise RuntimeError("type of axis value should be int")
    if op not in ReductionOp:
        raise RuntimeError("op can only be one of SUM, ASUM, SUMSQ , MEAN")
    if type(coeff) != int and type(coeff) != float:
        raise RuntimeError("coeff must be a value")
    axis_origin = axis
    shape_origin = shape
    axis = util.axis_check(len(shape), axis)
    util.check_reduce_shape_rule(shape)
    shape = list(shape)
    shape1 = shape[:axis] + [functools_reduce(lambda x, y: x*y, shape[axis:])]
    shape1, axis = util.shape_refine(shape1, axis)
    if len(axis) == 0:
        axis = [0]
        shape1 = [1] + shape1
    inp_dtype = dtype.lower()
    data = tvm.placeholder(shape1, name="data_input", dtype=inp_dtype)
    with tvm.target.cce():
        res = caffe_reduction_layer_compute([data], shape_origin, dtype,
                                            axis_origin, op, coeff, kernel_name,
                                            need_build, need_print)

    if op == "MEAN" and (inp_dtype == "int8" or inp_dtype == "uint8"):
        util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
        res = te.lang.cce.cast_to(res, inp_dtype)
        s = tvm.create_schedule(res.op)
        if need_print:
            with build_config:
                print(tvm.lower(s, [data, res], simple_mode=True))
        if need_build:
            with build_config:
                tvm.build(s, [data, res], "cce", name=kernel_name)
    else:
        with tvm.target.cce():
            sch = generic.auto_schedule(res)

        config = {"print_ir": need_print,
                  "need_build": need_build,
                  "name": kernel_name,
                  "tensor_list": [data, res]}
        te.lang.cce.cce_build_code(sch, config)
