"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf l2loss
"""

import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 200000000  # shape limit

@util.check_input_type((list, tuple), str, str, bool, bool)
def custom_l2_loss(shape, dtype, kernel_name="cce_tf_l2_loss", need_build=False, need_print=False):
    """
    Computes half the L2 norm of a tensor without the sqrt:
    output = sum(t ** 2) / 2
    
    Parameters
    ----------
    shape : shape of data

    dtype : source data type, only support float16, float32

    kernel_name : cce kernel name, default value is "cce_reductionLayer"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None
        
    """
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    util.check_reduce_shape_rule(shape)
    check_list = ["float16", "float32"]
    if not (dtype.lower() in check_list):
        raise RuntimeError(
            "tf_l2_loss_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    shape, axis = util.simplify_axis_shape(shape, range(len(shape)))

    inp_dtype = dtype.lower()
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)

    coeff_sqrt = tvm.const(1.0 / (2**(0.5)), dtype=inp_dtype)

    data_mul = te.lang.cce.vmuls(data_input, coeff_sqrt)
    data_sqr = te.lang.cce.vmul(data_mul, data_mul)
    res = te.lang.cce.sum(data_sqr, axis)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": need_print,
              "need_build": need_build,
              "name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
