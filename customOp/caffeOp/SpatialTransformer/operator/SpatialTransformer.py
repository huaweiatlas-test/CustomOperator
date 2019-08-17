"""
Copyright 2019 Huawei Technologies Co., Ltd

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
import topi
from te.platform.cce_build import build_config

def SpatialTransformer(input_shape, out_shape, dtype="float32", kernel_name="SpatialTransformer", need_build = True, need_print = False):
    """Spatial Transformer Layer
    
    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_.
    
    Parameters
    ----------
    input_shape : 
        the shape of input tensor
        [num_batch, height, width, num_channels]
        
    out_shape: float
        the height and width of output tensor
        [out_height, out_width].
        
    out_size: tuple of two ints
        The size of the output of the network (height, width)
        
    dtype: data type    
        
    kernel_name : kernel name, default value is "SpatialTransformer"

    need_buid : if need to build CCEC kernel, default value is True

    need_print : if need to print the ir, default value is False
    
    Returns
    -------
    tvm.Tensor

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
    .. [2]  https://github.com/tensorflow/models/tree/master/research/transformer
    """
    
    def _meshgrid(height, width):
		
        y0 = tvm.compute((height,), lambda i: -1 + i * 2.0 / (height - 1), name = 'y0')
        x0 = tvm.compute((width,), lambda i: -1 + i * 2.0 / (width - 1), name = 'x0')
        
        y = tvm.compute((height * width,), lambda i: y0[i // width], name = 'y')
        x = tvm.compute((height * width,), lambda i: x0[i % width], name = 'x')
        
        y = topi.reshape(y, (1, height * width))
        x = topi.reshape(x, (1, height * width))
        ones = tvm.compute((1, height * width), lambda i,j:1, name = 'ones')
         
        grid = tvm.compute((3, height * width),lambda i,j: 0.5 * (i - 1) * (i - 2) * x[0,j] + i * (2 - i) * y[0,j] + 0.5 * i * (i-1) * ones[0,j], name = 'grid')
        
        #grid = topi.concatenate((x,y,ones),0) #can not use topi.concatenate
        return grid       

    def _interpolate(im, im_shape, x, y, out_size, dtype):
        
        num_batch = im_shape[0]
        height = im_shape[1]
        width = im_shape[2]
        channels = im_shape[3]
            
        out_height = out_size[0]
        out_width = out_size[1]
        max_y = int(im_shape[1] - 1)
        max_x = int(im_shape[2] - 1)
               
        #[-1,1] -> [0, width-1]
        x = topi.multiply(topi.add(x, tvm.const(1, dtype=dtype)), width / tvm.const(2, dtype=dtype))
        y = topi.multiply(topi.add(y, tvm.const(1, dtype=dtype)), height / tvm.const(2, dtype=dtype))
            
        # do sampling
        dim3 = out_height * out_width * num_batch
            
        x0 = topi.cast(topi.floor(x), 'int32')  
        y0 = topi.cast(topi.floor(y), 'int32')
        x1 = topi.add(x0,tvm.const(1, dtype="int32"))
        y1 = topi.add(y0,tvm.const(1, dtype="int32"))

        x0 = topi.clip(x0, 0, max_x)
        x1 = topi.clip(x1, 0, max_x)
        y0 = topi.clip(y0, 0, max_y)
        y1 = topi.clip(y1, 0, max_y)

        dim2 = width
        dim1 = width * height

        base = tvm.compute((dim3,),lambda i:(i // (out_height * out_width)) * width * height, name = 'base')        
        base_y0 = topi.add(base, topi.multiply(y0, dim2))
        base_y1 = topi.add(base, topi.multiply(y1, dim2))

        idx_a = topi.add(base_y0, x0)
        idx_b = topi.add(base_y1, x0)
        idx_c = topi.add(base_y0, x1)
        idx_d = topi.add(base_y1, x1)
                
        im_flat = topi.reshape(im, (num_batch * height * width, channels))
        im_flat = topi.cast(im_flat, dtype)
        
        Ia = tvm.compute((dim3, channels),lambda i,j: im_flat[idx_a[i], j], name = 'Ia')       
        Ib = tvm.compute((dim3, channels),lambda i,j: im_flat[idx_b[i], j], name = 'Ib') 
        Ic = tvm.compute((dim3, channels),lambda i,j: im_flat[idx_c[i], j], name = 'Ic')
        Id = tvm.compute((dim3, channels),lambda i,j: im_flat[idx_d[i], j], name = 'Id')
            
        x0_f = topi.cast(x0, dtype)
        x1_f = topi.cast(x1, dtype)
        y0_f = topi.cast(y0, dtype)
        y1_f = topi.cast(y1, dtype)
        wa = topi.expand_dims(topi.multiply(topi.subtract(x1_f, x), topi.subtract(y1_f, y)), 1)
        wb = topi.expand_dims(topi.multiply(topi.subtract(x1_f, x), topi.subtract(y, y0_f)), 1)
        wc = topi.expand_dims(topi.multiply(topi.subtract(x, x0_f), topi.subtract(y1_f, y)), 1)
        wd = topi.expand_dims(topi.multiply(topi.subtract(x, x0_f), topi.subtract(y, y0_f)), 1)
 
        output = topi.add(topi.add(topi.add(topi.multiply(wa, Ia), topi.multiply(wb, Ib)),topi.multiply(wc, Ic)), topi.multiply(wd, Id))
        
        return output

    def _transform(theta, input_dim, out_size, input_shape, dtype):
        
        num_batch = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        num_channels = input_shape[3]

        theta = topi.reshape(theta, (num_batch, 2, 3))
        theta = topi.cast(theta, dtype)

        out_height = out_size[0]
        out_width = out_size[1]
                
        grid = _meshgrid(out_height, out_width)       
        grid = topi.reshape(grid, (num_batch, 3, out_height*out_width))
        grid = topi.cast(grid, dtype=dtype)
        
        k = tvm.reduce_axis((0, 3), 'k')
        T_g = tvm.compute((num_batch, 2, out_height*out_width),lambda b, y, x: tvm.sum(theta[b, y, k] * grid[b, k, x], axis = k), name = 'T_g')
              
        x_s = tvm.compute((num_batch, 1, out_height*out_width), lambda i,j,k:T_g[i,0,k], name = 'x_s')
        y_s = tvm.compute((num_batch, 1, out_height*out_width), lambda i,j,k:T_g[i,1,k], name = 'y_s')
              
        x_s_flat = topi.reshape(x_s, (num_batch*out_height*out_width,))
        y_s_flat = topi.reshape(y_s, (num_batch*out_height*out_width,))
                      
        input_transformed = _interpolate(input_dim, input_shape, x_s_flat, y_s_flat, out_size, dtype)
        output = topi.reshape(input_transformed, [num_batch, out_height, out_width, num_channels])
        return output 
    
    num_batch = input_shape[0]
    input_height = input_shape[1]
    input_width = input_shape[2]
    channel = input_shape[3]
    
    U = tvm.placeholder((num_batch, input_height, input_width, channel), name="U", dtype=dtype)    
    theta = tvm.placeholder((num_batch, 6, 1, 1), dtype=dtype)    
    output = _transform(theta, U, out_shape, input_shape, dtype)       
    s = tvm.create_schedule(output.op)

    if need_print:
        with build_config:
            print(tvm.lower(s, [U, theta, output], simple_mode=True))
            
    if need_build:
        with build_config:
            tvm.build(s, [U, theta, output], "cce", name=kernel_name)

    
if __name__ == "__main__":    
    out_size = (6, 8)
    SpatialTransformer((3, 6, 8, 3), out_size, "float16", need_build=True, need_print=True)
