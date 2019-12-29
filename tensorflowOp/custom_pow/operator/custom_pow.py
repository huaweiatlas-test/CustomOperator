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

tf pow
"""

from te import tvm

from te.platform.cce_build import build_config
from topi.cce import util

SHAPE_SIZE_LIMIT = 100000000  # shape limit for tf_pow


@util.check_input_type((list, tuple), (list, tuple), str, str, bool, bool)
def custom_pow(shape, shape_y, dtype, kernel_name="cce_tf_pow",
               need_build=False, need_print=False):
    """
    calculate x^y, calculating data type is float16 or float32 or int32
    when x < 0 , the output is a meaningless value.
    Parameters
    ----------
    shape : shape of data

    dtype : the data type, assume src_dtype equals dst_dtype, only support
    float16, float32, int32

    kernel_name : cce kernel name, default value is "tf_pow_cce"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None

    """
    supported_dtypes = ["float16", "float32", "int32"]
    device_api = "cc_device_pow"

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    if not dtype.lower() in supported_dtypes:
        raise RuntimeError("tf_pow_cce only support %s while dtype is %s"
                           % (",".join(supported_dtypes), dtype))

    inp_dtype = dtype.lower()
    shape = util.shape_refine(shape)
    data_lhs = tvm.placeholder(shape, name="data_lhs", dtype=inp_dtype)
    data_rhs = tvm.placeholder(shape, name="data_rhs", dtype=inp_dtype)

    v_datatype = util.get_device_api_dtype(inp_dtype)
    v_ndim = len(shape)
    block_num = "block_num"
    block_idx = "block_idx"
    pad_c0 = 0
    p_scale = util.create_param_ptr([0], inp_dtype, "p_scale")
    p_shift = util.create_param_ptr([0], inp_dtype, "p_shift")
    p_power = util.create_param_ptr([0], inp_dtype, "p_power")
    p_shape = util.create_param_ptr(shape, "int32", "p_shape")

    output = tvm.extern(shape, [data_lhs, data_rhs, p_scale, p_shift, p_power,
                                p_shape],
                        lambda ins, outs:
                        tvm.call_extern("int32_t", device_api,
                                        block_num,
                                        block_idx,
                                        v_datatype,
                                        ins[2].access_ptr("r"),  # scale
                                        ins[3].access_ptr("r"),  # shift
                                        ins[4].access_ptr("r"),  # power
                                        v_ndim,
                                        ins[5].access_ptr("r"),  # shape
                                        pad_c0,
                                        ins[0].access_ptr("r"),  # input x
                                        v_ndim,
                                        v_ndim,
                                        ins[5].access_ptr("r"),  # shape
                                        pad_c0,
                                        ins[1].access_ptr("r"),  # input y
                                        outs[0].access_ptr("w")),
                        name="output", dtype=inp_dtype)

    schedule = tvm.create_schedule(output.op)

    if need_print:
        with build_config:
            print(tvm.lower(schedule, [data_lhs, data_rhs, output],
                            simple_mode=True))
    if need_build:
        with build_config:
            tvm.build(schedule, [data_lhs, data_rhs, output], "cce",
                      name=kernel_name)
