"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf log
"""

from te import tvm

from te.platform.cce_build import build_config

from topi.cce import util

SHAPE_SIZE_LIMIT =  200000000   # shape limit for tf_log

@util.check_input_type((list, tuple), str, str, bool, bool)
def custom_log(shape, dtype, kernel_name="cce_tf_log", need_build=False, need_print=False):
    """
    calculate ln(data), calculating data type is float16 or float32

    Parameters
    ----------
    shape : shape of data

    dtype : source data type, assume src_dtype equals dst_type, only support
        float16 or float32

    kernel_name : cce kernel name, default value is "cce_tf_log"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None

    """
    supported_dtypes = ["float16", "float32"]
    device_api = "DeviceLog"

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    if not (dtype.lower() in supported_dtypes):
        raise RuntimeError(
            "tf_log_cce only support %s while dtype is %s" % (",".join(supported_dtypes), dtype))

    inp_dtype = dtype.lower()
    shape = util.shape_refine(shape)
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)

    v_datatype = util.get_device_api_dtype(inp_dtype)
    v_ndim = len(shape)
    block_num = "block_num"
    block_idx = "block_idx"
    padC0 = 0
    p_scale = util.create_param_ptr([1], inp_dtype, "p_scale")
    p_shift = util.create_param_ptr([0], inp_dtype, "p_shift")
    p_base = util.create_param_ptr([-1], inp_dtype, "p_base")
    p_shape = util.create_param_ptr(shape, "int32", "p_shape")

    output = tvm.extern(shape, [data_input, p_scale, p_shift, p_base, p_shape], lambda ins, outs:
    tvm.call_extern("int32_t", device_api,
                    block_num,
                    block_idx,
                    v_datatype,
                    ins[1].access_ptr("r"),  # scale
                    ins[2].access_ptr("r"),  # shift
                    ins[3].access_ptr("r"),  # base
                    v_ndim,
                    ins[4].access_ptr("r"),  # shape
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
