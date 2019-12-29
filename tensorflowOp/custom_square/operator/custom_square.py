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

tf square
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 200000000  # shape limit


@fusion_manager.register("custom_square_cce")
def custom_square_compute(placeholders, shape, dtype,
                          kernel_name="cce_custom_square",
                          need_build=False,
                          need_print=False):
    """
        Parameters
        ----------
        shape : shape of data
        dtype : the data type, assume src_dtype equals dst_dtype,
            only support float16, float32, int32
        kernel_name : cce kernel name, default value is "cce_custom_square"
        need_buid : if need to build CCEC kernel, default value is False
        need_print : if need to print the ir, default value is False
        Returns
        -------
        None
    """
    data = placeholders[0]

    res = te.lang.cce.vmul(data, data)
    return res


@util.check_input_type((list, tuple), str, str, bool, bool)
def custom_square(shape, dtype, kernel_name="cce_custom_square",
                  need_build=False,
                  need_print=False):
    """
    algorithm: custom_square
    calculating data's custom_square,y= x*x
    Parameters
    ----------
    shape : shape of data
    dtype : the data type, assume src_dtype equals dst_dtype,
            only support float16, float32, int32
    kernel_name : cce kernel name, default value is "cce_custom_square"
    need_buid : if need to build CCEC kernel, default value is False
    need_print : if need to print the ir, default value is False
    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    check_list = ["float16", "float32", "int32"]
    if not dtype.lower() in check_list:
        raise RuntimeError(
            "custom_square_cce only support %s while dtype is %s"
            % (",".join(check_list), dtype))

    shape = util.shape_refine(shape)
    data = tvm.placeholder(shape, name="data", dtype=dtype.lower())

    with tvm.target.cce():
        res = custom_square_compute([data], shape, dtype, kernel_name,
                                    need_build, need_print)
        sch = generic.auto_schedule(res)

    config = {"print_ir": need_print,
              "need_build": need_build,
              "name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)
