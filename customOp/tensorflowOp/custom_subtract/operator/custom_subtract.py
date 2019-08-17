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
from topi import generic

from topi.cce import util

SHAPE_SIZE_LIMIT = 200000000  # shape limit

@util.check_input_type((list, tuple), (list, tuple), str, str, bool, bool)
def custom_subtract(shape_x, shape_y, dtype, kernel_name="cce_subtract", need_build=True,
                    need_print=True):
    """
    do element-wise subtract operation between two input tensors

    Parameters:
    ----------
    shape_x : shape of input data1

    shape_y : shape of input data2

    dtype : source data type, support float16,float32,int32

    kernel_name : cce kernel name, default value is "cce_subtract"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)

    check_list = ["float16", "float32", "int32"]
    dtype = dtype.lower()
    if not (dtype in check_list):
        raise RuntimeError(
            "tf_subtract_cce only support %s while dtype is %s" % (",".join(check_list), dtype))
    print("######## shape")
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)

    data1 = tvm.placeholder(shape_x, dtype=dtype, name="data1")
    data2 = tvm.placeholder(shape_y, dtype=dtype, name="data2")

    with tvm.target.cce():
        data1_tmp1 = te.lang.cce.broadcast(data1, shape_max)
        data2_tmp1 = te.lang.cce.broadcast(data2, shape_max)
        res = te.lang.cce.vsub(data1_tmp1, data2_tmp1)
        sch = generic.auto_schedule(res)

    config = {"print_ir": need_print,
              "need_build": need_build,
              "name": kernel_name,
              "tensor_list": [data1, data2, res]}
    te.lang.cce.cce_build_code(sch, config)
