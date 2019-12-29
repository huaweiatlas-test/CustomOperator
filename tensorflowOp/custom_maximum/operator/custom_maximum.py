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

tf maximum
"""
import te.lang.cce
from te import tvm

from topi import generic

from topi.cce import util

SHAPE_SIZE_LIMIT = 200000000  # shape limit


@util.check_input_type((list, tuple), (list, tuple), str, str, bool, bool)
def custom_maximum(shape1, shape2, dtype, kernel_name="cce_tf_maximum",
                   need_build=False,
                   need_print=False):
    """
    do element-wise maximum operation between two input tensors

    Parameters:
    ----------
    shape1 : shape of input data1

    shape2 : shape of input data2

    dtype : source data type, support float16,float32,int32

    kernel_name : cce kernel name, default value is "cce_tf_maximum"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None
    """

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape1)
    util.check_shape_rule(shape2)
    util.check_shape_size(shape1, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape2, SHAPE_SIZE_LIMIT)

    check_list = ["float16", "float32", "int32"]

    dtype = dtype.lower()
    if dtype not in check_list:
        raise RuntimeError(
            "tf_maximum_cce only support %s while dtype is %s" % (
                ",".join(check_list), dtype))

    shape1, shape2, shape_max = util.produce_shapes(shape1, shape2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)

    data1 = tvm.placeholder(shape1, dtype=dtype, name="data1")
    data2 = tvm.placeholder(shape2, dtype=dtype, name="data2")

    with tvm.target.cce():
        data1_tmp1 = te.lang.cce.broadcast(data1, shape_max)
        data2_tmp1 = te.lang.cce.broadcast(data2, shape_max)
        res = te.lang.cce.vmax(data1_tmp1, data2_tmp1)

        sch = generic.auto_schedule(res)

    config = {"print_ir": need_print,
              "need_build": need_build,
              "name": kernel_name,
              "tensor_list": [data1, data2, res]}
    te.lang.cce.cce_build_code(sch, config)
