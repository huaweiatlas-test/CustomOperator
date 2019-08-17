"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf sqrt
"""

from te import tvm
from topi.cce import util
from te.platform.cce_build import build_config

SHAPE_SIZE_LIMIT =  160000000  # shape limit for tf_sqrt

@util.check_input_type((list, tuple), str, str, bool, bool)
def custom_sqrt(shape, dtype, kernel_name="cce_sqrt", need_build=False, need_print=False):
    """
    calculate data ** 0.5, calulation method exp(0.5*log(data)), calculating data type is float16
    
    Parameters
    ----------
    shape : shape of data

    dtype : the data type, assume src_dtype equals dst_dtype, only support float16, float32

    kernel_name : cce kernel name, default value is "cce_sqrt"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None
        
    """
    check_list = ["float16", "float32"]
    device_api_map = {"float16": "cc_device_sqrt_float16",
                      "float32": "cc_device_sqrt_float"
                      }
    max_dim = 8
    shape_len = len(shape)
    if shape_len > max_dim:
        raise RuntimeError(
            "sqrt_cce only support up to %d dimensions while the shape's dimension is %d" %
            (max_dim, shape_len))

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    if not (dtype.lower() in check_list):
        raise RuntimeError(
            "sqrt_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    inp_dtype = dtype.lower()
    shape = util.shape_refine(shape)
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)
    device_api = device_api_map[inp_dtype]

    block_num = "block_num"
    block_idx = "block_idx"
    v_ndim = tvm.const(len(shape), "int32")
    padC0 = tvm.const(0, "int32")
    p_shape = util.create_param_ptr(shape, "int32", "p_shape")

    output = tvm.extern(shape, [data_input, p_shape], lambda ins, outs:
    tvm.call_extern("int32_t", device_api,
                    block_num,
                    block_idx,
                    v_ndim,
                    ins[1].access_ptr("r"),  # shape
                    padC0,
                    ins[0].access_ptr("r"),  # input x
                    outs[0].access_ptr("w")),
                        name="output", dtype=inp_dtype)

    s = tvm.create_schedule(output.op)

    if need_print:
        with build_config:
            print(tvm.lower(s, [data_input, output], simple_mode=True))
    if need_build:
        with build_config:
            tvm.build(s, [data_input, output], "cce", name=kernel_name)
