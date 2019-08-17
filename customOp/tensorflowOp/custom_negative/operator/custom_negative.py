"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf negative
"""

from topi.cce import util
from topi.cce import caffe2_negative
from te.platform.fusion_manager import fusion_manager

SHAPE_SIZE_LIMIT = 200000000  # shape limit

@fusion_manager.register("custom_negative_cce")
def custom_negative_compute(placeholders, shape, dtype,
                        kernel_name="cce_custom_negative", need_build=False,
                        need_print=False):
    return caffe2_negative.caffe2_negative_compute(placeholders, shape, dtype,
                                                   kernel_name, need_build,
                                                   need_print)


@util.check_input_type((list, tuple), str, str, bool, bool)
def custom_negative(shape, dtype, kernel_name="cce_custom_negative",
                    need_build=False, need_print=False):
    """
    calculate y = -x, calculating data type is float16
    
    Parameters
    ----------
    shape : shape of data

    dtype : the data type, assume src_dtype equals dst_dtype,
            only support float16, float32, int32

    kernel_name : cce kernel name, default value is "cce_custom_negative"

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
    if not (dtype.lower() in check_list):
        raise RuntimeError(
            "sqrt_cce only support %s while dtype is %s"
            % (",".join(check_list), dtype))

    caffe2_negative.caffe2_negative_cce(shape, dtype, kernel_name=kernel_name,
                                        need_build=need_build,
                                        need_print=need_print)
