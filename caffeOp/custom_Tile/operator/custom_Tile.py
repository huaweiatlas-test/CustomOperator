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

caffe tile layer
"""

from topi.cce import tf_tile
from topi.cce import util


@util.check_input_type((list, tuple), str, int, int, str, bool, bool)
def custom_Tile(shape, dtype, tiles, axis=1,
                kernel_name="cce_caffe_tile_layer",
                need_build=False, need_print=False):
    """Operation and Schedule for tilelayer, construct an array by axis and
    tiles.

    Parameters
    ----------
    shape: shape of Tensor

    dtype: the data type. only support float16, float32, int32, int8, uint8

    tiles: the number of copies (tiles) of the tensor to output

    axis: the index of the axis to tile

    kernel_name: cce kernel name, default value is "cce_caffe_tile_layer"

    need_buid: if need to build CCEC kernel, default value is False

    need_print: if need to print the ir, default value is False

    Returns
    -------
    None
    """
    check_list = ["float16", "float32", "int32", "int8", "uint8"]
    if not (dtype.lower() in check_list):
        raise RuntimeError(
            "caffe_tile_layer only support %s while dtype is %s" % (
                ",".join(check_list), dtype))

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)

    if not isinstance(axis, int):
        raise RuntimeError("type of axis value should be int")
    if axis >= len(shape) or axis < -len(shape):
        raise RuntimeError(
            "input axis is out of range, axis value can be from %d to %d" % (
                -len(shape), len(shape) - 1))

    if not isinstance(tiles, int):
        raise RuntimeError("type of tiles must be int.")
    if tiles < 0:
        raise RuntimeError("Number of tiles must be positive.")

    multiples = [1] * len(shape)

    multiples[axis] = tiles

    tf_tile.tf_tile_cce(shape, dtype, multiples, kernel_name=kernel_name,
                        need_build=need_build,
                        need_print=need_print)
