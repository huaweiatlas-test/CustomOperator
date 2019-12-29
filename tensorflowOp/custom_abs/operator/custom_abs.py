"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf abs
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 200000000  # shape limit


@fusion_manager.register("custom_abs_cce")
def custom_abs_compute(placeholders, shape, dtype,
                       kernel_name="cce_custom_abs",
                       need_build=False, need_print=False):
    """
        algorithm: custom_abs
        calculating data's abs,y= |x|
        Parameters
        ----------
        shape : shape of data
        dtype : the data type, assume src_dtype equals dst_dtype,
        only support float16, float32, int32
        kernel_name : cce kernel name, default value is "cce_custom_abs"
        need_buid : if need to build CCEC kernel, default value is False
        need_print : if need to print the ir, default value is False
        Returns
    -------
    None
    """
    data = placeholders[0]
    inp_dtype = dtype.lower()

    res = te.lang.cce.vabs(data)
    if inp_dtype == "int32":
        res = te.lang.cce.round(res)
    return res


@util.check_input_type((list, tuple), str, str, bool, bool)
def custom_abs(shape, dtype, kernel_name="cce_custom_abs",
               need_build=False, need_print=False):
    """
    algorithm: custom_abs
    calculating data's abs,y= |x|
    Parameters
    ----------
    shape : shape of data
    dtype : the data type, assume src_dtype equals dst_dtype,
    only support float16, float32, int32
    kernel_name : cce kernel name, default value is "cce_custom_abs"
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
            "abs_cce only support %s while dtype is %s"
            % (",".join(check_list), dtype))

    inp_dtype = dtype.lower()
    shape = util.shape_refine(shape)
    data = tvm.placeholder(shape, name="data", dtype=inp_dtype)

    with tvm.target.cce():
        res = custom_abs_compute([data], shape, dtype, kernel_name, need_build,
                                 need_print)
        sch = generic.auto_schedule(res)

    config = {"print_ir": need_print,
              "need_build": need_build,
              "name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)
