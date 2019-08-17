"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

caffe pow layer
"""
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util

from numpy import float16 as np_fp16
from numpy import float32 as np_fp32

SHAPE_SIZE_LIMIT =  100000000   # shape limit for caffe_power

@util.check_input_type((list, tuple), str, (int,float,np_fp16,np_fp32), (int,float,np_fp16,np_fp32),
                       (int,float,np_fp16,np_fp32), str, bool, bool)
def custom_Power(shape, dtype, gamma, alpha, beta, kernel_name="cce_caffe_power",
                          need_build=False, need_print=False):
    """
    calculate (alpha * data + beta) ** gamma, calulation method exp(gamma * log(alpha * data + beta)).
    when alpha * data + beta < 0 , the output is a meaningless value.
    Parameters
    ----------
    shape : shape of data

    dtype : the data type, assume src_dtype equals dst_dtype, only support float16, float32

    gamma : the data type must be same with dtype parameter
        args in (alpha * data + beta) ** gamma

    alpha : the data type must be same with dtype parameter
        args in (alpha * data + beta) ** gamma

    beta : the data type must be same with dtype parameter
        args in (alpha * data + beta) ** gamma

    kernel_name : string
        kernel name in generated CCE kernal. default value is "cce_caffe_power"


    need_buid : bool
        if need to build CCEC kernel

    need_print : bool
        if need to print Halide IR

    Returns
    -------
    None
        
    """
    supported_dtypes = ["float16", "float32"]
    device_api = "cc_device_pow"

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    if not (dtype.lower() in supported_dtypes):
        raise RuntimeError(
            "power_cce only support %s while dtype is %s" % (",".join(supported_dtypes), dtype))

    inp_dtype = dtype.lower()
    shape = util.shape_refine(shape)
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)

    v_datatype = util.get_device_api_dtype(inp_dtype)
    v_ndim_x = len(shape)
    v_ndim_y = 0
    p_shape_y = 0
    p_input_y = "nullptr"
    block_num = "block_num"
    block_idx = "block_idx"
    padC0 = 0

    p_scale = util.create_param_ptr([alpha], inp_dtype, "p_scale")
    p_shift = util.create_param_ptr([beta], inp_dtype, "p_shift")
    p_power = util.create_param_ptr([gamma], inp_dtype, "p_power")
    p_shape_x = util.create_param_ptr(shape, "int32", "p_shape_x")

    # scale --> alpha, shitf --> beta, power --> gamma
    output = tvm.extern(shape, [data_input, p_scale, p_shift, p_power, p_shape_x], lambda ins, outs:
    tvm.call_extern("int32_t", device_api,
                    block_num,
                    block_idx,
                    v_datatype,
                    ins[1].access_ptr("r"),  # scale
                    ins[2].access_ptr("r"),  # shift
                    ins[3].access_ptr("r"),  # power
                    v_ndim_x,
                    ins[4].access_ptr("r"),  # shape
                    padC0,
                    ins[0].access_ptr("r"),  # input x
                    v_ndim_y,
                    v_ndim_y,
                    p_shape_y,
                    padC0,
                    p_input_y,
                    outs[0].access_ptr("w")),
                        name="output", dtype=inp_dtype)

    s = tvm.create_schedule(output.op)

    if need_print:
        with build_config:
            print(tvm.lower(s, [data_input, output], simple_mode=True))
    if need_build:
        with build_config:
            tvm.build(s, [data_input, output], "cce", name=kernel_name)
