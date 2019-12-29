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

caffe log layer
"""

from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util

SHAPE_SIZE_LIMIT = 200000000  # shape limit for caffe_log


@util.check_input_type((list, tuple), str, (int, float), (int, float),
                       (int, float), str, bool, bool)
def custom_Log(shape, dtype, gamma=-1, alpha=1, beta=0,
               kernel_name="cce_caffe_log_layer",
               need_build=False, need_print=False):
    """
    calculate y = log(gamma, alphax+beta) = log(alphax+beta) / log(gamma) for
    gamma > 0, calculating data type is float16 or float32

    Parameters
    ----------
    shape : shape of data

    dtype : the data type, assume src_dtype equals dst_dtype, only support
    float16, float32

    gamma : the base, default value is -1 for "math.e", gamma

    alpha : the scale, default value is 1

    beta : the shift, default value is 0

    kernel_name : cce kernel name, default value is "cce_caffe_log_layer"

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

    if not dtype.lower() in supported_dtypes:
        raise RuntimeError(
            "caffe_log_layer_cce only support %s while dtype is %s" % (
                ",".join(supported_dtypes), dtype))

    if gamma != -1 and gamma <= 0:
        # api  cc_device_exp_c handle gamma == -1 as e
        raise ValueError(
            "please ensure gamma is greater than 0, where gamma = %s" % str(
                gamma))

    inp_dtype = dtype.lower()
    shape = util.shape_refine(shape)
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)

    v_datatype = util.get_device_api_dtype(inp_dtype)
    v_ndim = len(shape)
    block_num = "block_num"
    block_idx = "block_idx"
    pad_c0 = 0
    p_scale = util.create_param_ptr([alpha], inp_dtype, "p_scale")
    p_shift = util.create_param_ptr([beta], inp_dtype, "p_shift")
    p_base = util.create_param_ptr([gamma], inp_dtype, "p_base")
    p_shape = util.create_param_ptr(shape, "int32", "p_shape")

    # scale --> alpha, shitf --> beta, base --> gamma
    output = tvm.extern(shape, [data_input, p_scale, p_shift, p_base, p_shape],
                        lambda ins, outs:
                        tvm.call_extern("int32_t", device_api,
                                        block_num,
                                        block_idx,
                                        v_datatype,
                                        ins[1].access_ptr("r"),  # scale
                                        ins[2].access_ptr("r"),  # shift
                                        ins[3].access_ptr("r"),  # base
                                        v_ndim,
                                        ins[4].access_ptr("r"),  # shape
                                        pad_c0,
                                        ins[0].access_ptr("r"),  # input x
                                        outs[0].access_ptr("w")),
                        name="output", dtype=inp_dtype)

    schedule = tvm.create_schedule(output.op)

    if need_print:
        with build_config:
            print(tvm.lower(schedule, [data_input, output], simple_mode=True))
    if need_build:
        with build_config:
            tvm.build(schedule, [data_input, output], "cce", name=kernel_name)
