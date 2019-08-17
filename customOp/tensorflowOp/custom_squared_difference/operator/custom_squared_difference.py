"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf squared_difference
"""

import te.lang.cce
from te import tvm
from topi import generic

from topi.cce import util

SHAPE_SIZE_LIMIT = 200000000  # shape limit

@util.check_input_type((list, tuple), (list, tuple), str, str, bool, bool)
def custom_squared_difference(shape_x, shape_y, dtype, kernel_name="cce_tf_squared_difference", need_build=False, need_print=False):
    """
    algorithm: tf_squared_difference

    calculating data's tf_squared_difference,y= (x - y) * (x - y)

    Parameters
    ----------
    shape_x : shape of input x

    shape_y : shape of input y

    dtype : the data type, assume src_dtype equals dst_dtype, only support float16, float32, int32

    kernel_name : cce kernel name, default value is "cce_tf_squared_difference"

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

    check_list = ["float16", "float32", "int32"]

    if not (dtype.lower() in check_list):
        raise RuntimeError("tf_squared_difference_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    dtype = dtype.lower()

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)

    data_x = tvm.placeholder(shape_x, dtype=dtype, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype, name="data_y")

    with tvm.target.cce():
        data_x_tmp = te.lang.cce.broadcast(data_x, shape_max)
        data_y_tmp = te.lang.cce.broadcast(data_y, shape_max)
        data_sub = te.lang.cce.vsub(data_x_tmp, data_y_tmp)
        res = te.lang.cce.vmul(data_sub, data_sub)
        sch = generic.auto_schedule(res)

    config = {"print_ir" : need_print,
              "need_build" : need_build,
              "name" : kernel_name,              
              "tensor_list" : [data_x, data_y, res]}

    te.lang.cce.cce_build_code(sch, config)
