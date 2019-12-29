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

tf sign
"""

from te import tvm
from topi import generic

import te.lang.cce
from topi.cce import util
from te.platform.fusion_manager import fusion_manager

SHAPE_SIZE_LIMIT = 200000000  # shape limit


@fusion_manager.register("custom_sign_cce")
def custom_sign_compute(placeholders, shape, dtype,
                        kernel_name="cce_custom_sign",
                        need_build=False, need_print=False):
    """
    compute for custom_sign_cce
    """
    data = placeholders[0]
    inp_dtype = dtype.lower()
    fp16_max = tvm.const(32768, dtype=inp_dtype)
    fp16_min = tvm.const(2 ** (-15), dtype=inp_dtype)
    data_tmp = data
    if dtype == "float16":
        data_tmp = te.lang.cce.round_to(data, 0.5, -0.5)

    new_data = te.lang.cce.vmuls(data_tmp, fp16_max)
    tmp2 = te.lang.cce.vabs(new_data)
    anuminate = te.lang.cce.vadds(tmp2, fp16_min)
    rec = te.lang.cce.vrec(anuminate)
    fp16_res = te.lang.cce.vmul(new_data, rec)
    res = te.lang.cce.round(fp16_res)
    return res


@util.check_input_type((list, tuple), str, str, bool, bool)
def custom_sign(shape, dtype, kernel_name="cce_custom_sign", need_build=False,
                need_print=False):
    """
                                  x*32768
    algrithm: sign = round(-------------------------)
                            2 ** (-15) + |x*32768|

    calculating data type is float16

    Parameters
    ----------
    shape : shape of data

    dtype : the data type, assume src_dtype equals dst_dtype,
            only support float16, float32, int32

    kernel_name : cce kernel name, default value is "cce_sign"

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
        raise RuntimeError("custom_sign_cce only support %s while dtype is %s"
                           % (",".join(check_list), dtype))

    shape = util.shape_refine(shape)
    inp_dtype = dtype.lower()
    data = tvm.placeholder(shape, name="data", dtype=inp_dtype)
    with tvm.target.cce():
        res = custom_sign_compute([data], shape, dtype,
                                  kernel_name, need_build, need_print)

        sch = generic.auto_schedule(res)

    config = {"print_ir": need_print,
              "need_build": need_build,
              "name": kernel_name,
              "tensor_list": [data, res]}
    te.lang.cce.cce_build_code(sch, config)
