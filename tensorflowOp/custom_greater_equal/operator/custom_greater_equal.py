"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import te.lang.cce
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util

SHAPE_SIZE_LIMIT = 250000000  # shape limit for tf_greater_equal


@util.check_input_type((list, tuple), (list, tuple), str, str, bool, bool)
def custom_greater_equal(shape_x, shape_y, dtype,
                         kernel_name="cce_tf_greater_equal",
                         need_build=False, need_print=False):
    """
    do element-wise greater equal operation between two input tensors

    Parameters:
    ----------
    shape_x : shape of input data1

    shape_y : shape of input data2

    dtype : source data type, support [float16,float32,int32,int8,uint8]

    kernel_name : cce kernel name, default value is "cce_tf_greater_equal"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None
    """

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)

    check_list = ["float16", "float32", "int32", "int8", "uint8", "bool"]

    dtype = dtype.lower()
    if dtype not in check_list:
        raise RuntimeError(
            "tf_equal_cce only support %s while dtype is %s" % (
                ",".join(check_list), dtype))

    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)

    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)

    data1 = tvm.placeholder(shape_x, dtype=dtype, name="data1")
    data2 = tvm.placeholder(shape_y, dtype=dtype, name="data2")

    data1_tmp1 = te.lang.cce.broadcast(data1, shape_max)
    data2_tmp1 = te.lang.cce.broadcast(data2, shape_max)

    res = tvm.compute(shape_max, lambda *i: data1_tmp1(*i) >= data2_tmp1(*i),
                      name='res')

    sch = tvm.create_schedule(res.op)

    if need_print:
        with build_config:
            print(tvm.lower(sch, [data1, data2, res], simple_mode=True))

    if need_build:
        with build_config:
            tvm.build(sch, [data1, data2, res], "cce", name=kernel_name)
