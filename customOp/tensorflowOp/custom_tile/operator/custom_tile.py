# pylint: disable=invalid-name, too-many-locals, too-many-statements
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf tile
"""

from te import tvm

from te.platform.cce_build import build_config
from topi.nn.tile import compute_tile_cce
from topi.cce import util

SHAPE_SIZE_LIMIT = 20000000  # shape limit for tf_tile

def schedule_tile_cce(out):
    """Schedule for cce tile arithmetic operator.

    Parameters
    ----------
    out: TVM Tensor
        The computation graph description of cce tile.

    Returns
    -------
    s: Schedule
        The computation schedule for tile.
    """
    sch = tvm.create_schedule(out.op)
    return sch

@util.check_input_type((list, tuple), (list, tuple), str, str, bool, bool)
def custom_tile(shape, multiples, dtype, kernel_name="cce_tile", need_build=False,
                need_print=False):
    """Operation and Schedule for tile, construct an array by repeating shape the number of times given by multiply_shape.

    Parameters
    ----------
    shape:shape of Tensor
    
    multiples:  shape of Tensor
    
    dtype: 
        the data type. only support float16, float32, int32, int8, uint8

    kernel_name : cce kernel name, default value is "cce_tile"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
        None
    """
    check_list = ["float16", "float32", "int32", "int8", "uint8"]
    if not (dtype.lower() in check_list):
        raise RuntimeError(
            "tile_cce only support %s while dtype is %s" % (",".join(check_list), dtype))
    tensor_l = []

    inp_dtype = dtype.lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    tensor_l.append(tvm.placeholder(shape, name="shape", dtype=inp_dtype))

    for i in range(len(multiples)):
        if not isinstance(multiples[i], int):
            raise RuntimeError(
                "InvalidArgumentError: Expected int value")
        if multiples[i] < 0:
            raise RuntimeError(
                "InvalidArgumentError: Expected int value or multiples[%d] >= 0, but got %d!" % (
                i, multiples[i]))

    tensor_l.append(tvm.placeholder(multiples, name="multiples", dtype=inp_dtype))

    out_tensor = compute_tile_cce(a_tuple=tensor_l)

    s = schedule_tile_cce(out_tensor)
    if need_print:
        with build_config:
            print(tvm.lower(s, [tensor_l[0], tensor_l[1], out_tensor], simple_mode=True))

    if need_build:
        with build_config:
            tvm.build(s, tensor_l + [out_tensor], "cce", name=kernel_name)
