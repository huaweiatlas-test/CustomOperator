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

from te import tvm
from topi.cce import util
from te.platform.cce_build import build_config

SHAPE_SIZE_LIMIT =  150000000  # shape limit for tf_logical_not

@util.check_input_type((list, tuple), str, str, bool, bool)
def custom_logical_not(shape, dtype, kernel_name="cce_tf_logical_not", need_build=False,
                       need_print=False):
    """
    logical not for the input tensor
    
    Parameters
    ----------
    shape : input shape of data

    dtype : the data type, support bool

    kernel_name : cce kernel name, default value is "cce_logical_not"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None
        
    """
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)

    check_list = ["bool"]
    if not (dtype.lower() in check_list):
        raise RuntimeError(
            "logical_not_cce ony supports %s while dtype is %s" % (",".join(check_list), dtype))

    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    inp_dtype = dtype.lower()

    data = tvm.placeholder(shape, name="data", dtype=inp_dtype)

    with tvm.target.cce():

        result = tvm.compute(shape, lambda *i: tvm.select(data[i] == True, False, True),
                             name="result")

        s = tvm.create_schedule(result.op)

        if need_print:
            with build_config:
                print(tvm.lower(s, [data, result], simple_mode=True))
        if need_build:
            with build_config:
                tvm.build(s, [data, result], "cce", name=kernel_name)
