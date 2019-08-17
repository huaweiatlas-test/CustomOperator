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
import sys
import numpy as np

from spatial_transformer import transformer
import tensorflow as tf


def dump_data(input_data, name, fmt, data_type):
    if fmt == "binary" or fmt == "bin":
        f_output = open(name, "wb")
        if (data_type == "float16"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.float16(elem).tobytes())
        elif (data_type == "float32"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.float32(elem).tobytes())
        elif (data_type == "int32"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.int32(elem).tobytes())
        elif (data_type == "int8"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.int8(elem).tobytes())
        elif (data_type == "uint8"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.uint8(elem).tobytes())
    else:
        f_output = open(name, "w")
        index = 0
        for elem in np.nditer(input_data):
            f_output.write("%f\t" % elem)
            index += 1
            if index % 16 == 0:
                f_output.write("\n")
    
def gen_STN():
    #x = np.reshape(np.arange(5*5*1),(1,5,5,1)).astype(np.float32)
    sess = tf.Session()
    U = tf.range(3*6*8*3)
    U = tf.reshape(U,[3, 6, 8, 3])#input is NCHW

    theta = tf.constant([[1, 0, 0, 0, 1, 0],[1, 0, 0, 0, 1, 0],[1, 0, 0, 0, 1, 0]])
    output = transformer(U, theta, (6,8))
    
    
    dump_data(sess.run(U), "U.data", fmt="binary",data_type="float32")              
    dump_data(sess.run(U), "U.txt", fmt="float",data_type="float32") 

    dump_data(sess.run(theta), "theta.data", fmt="binary",data_type="float32")              
    dump_data(sess.run(theta), "theta.txt", fmt="float",data_type="float32")

    dump_data(sess.run(output), "output.data", fmt="binary",data_type="float32")              
    dump_data(sess.run(output), "output.txt", fmt="float",data_type="float32")      

if __name__ == "__main__":
    #sign("sign")
    #gen_reduction_data("Reduction", "SUM", 1, 2)
    gen_STN()