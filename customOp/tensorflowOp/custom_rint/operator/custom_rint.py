"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf rint
"""
from topi.cce import util
from topi.cce import tf_round

@util.check_input_type((list, tuple), str, str, bool, bool)
def custom_rint(shape, dtype, kernel_name="cce_tf_rint", need_build=False, need_print=False):
    """
    calculate rint(data), calculating data type is float16 or float32

    Parameters
    ----------
    shape : shape of data

    dtype : source data type, assume src_dtype equals dst_type, only support 
        float16 or float32

    kernel_name : cce kernel name, default value is "cce_tf_rint"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None

    """
    max_dim = 8
    shape_len = len(shape)
    if shape_len > max_dim:
        raise RuntimeError(
            "rint_cce only support up to %d dimensions while the shape's dimension is %d" %
            (max_dim, shape_len))

    tf_round.tf_round_cce(shape, dtype, kernel_name=kernel_name, need_build=need_build,
                          need_print=need_print)
