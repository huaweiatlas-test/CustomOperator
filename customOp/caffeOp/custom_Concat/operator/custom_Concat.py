"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf concact op
"""

import te.lang.cce

from te import tvm
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce

@util.check_input_type((list, tuple), str, int, str, bool, bool)
def custom_Concat(shapes, dtype, axis, kernel_name="concat", need_build=False, need_print=False):
    """
    concat one or two input data

    Parameters
    ----------
    shapes : input shape of data

    dtype : the data type, assume src_dtype equals dst_dtype, support uint8, int8, int32, float16, float32

    axis : concat axis

    kernel_name : cce kernel name, default value is "concat"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None

    """

    
    util.check_kernel_name(kernel_name)

    for i in range(len(shapes)):
        util.check_shape_rule(shapes[i])

    sum_dim = 0
    for shape in shapes:
        sum_dim += functools_reduce(lambda x, y: x*y, shape)

    if sum_dim > 2**31-1:
        raise RuntimeError("shape exceed 32bit limitation")

    check_list = ["uint8", "int8", "float16", "float32", "int32"]
    if not (dtype.lower() in check_list):
        raise RuntimeError(
            "concat_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    inp_dtype = dtype.lower()
    data = []
    for i in range(len(shapes)):
        shape = shapes[i]
        data.append(tvm.placeholder(shape, name="data_%d" % i, dtype=inp_dtype))

    with tvm.target.cce():
        res = te.lang.cce.concat(data, axis)
        sch = generic.auto_schedule(res)

    data.append(res)

    config = {"print_ir": need_print,
              "need_build": need_build,
              "name": kernel_name,
              "tensor_list": data}

    te.lang.cce.cce_build_code(sch, config)
