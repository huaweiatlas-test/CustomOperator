**Interface**

```python
def SpatialTransformer(
    input_shape, 
    out_shape, 
    dtype="float32", 
    kernel_name="SpatialTransformer", 
    need_build = True, 
    need_print = False)
```

**Description**

Implements a spatial transformer layer as described in the paper *[Spatial Transformer Networks](<https://arxiv.org/abs/1506.02025>)*.

Based on the [tensorflow/models github code](<https://github.com/tensorflow/models/tree/master/research/transformer>).

**Args:**

- input_shape: the shape of input tensor, (num_batch, height, width, num_channels).
- out_shape: the height and width of output tensor , (out_height, out_width).
- out_size: the size of the output of the network, (height, width).
- dtype: support `float16` and `float32`.
- kernel_name: op's kernel func name, optional
- need_build: whether build CCEC kernel, default is `True`, optional
- need_print: whether print IR, default is `False`, optional

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`

**Notice**

1. Before plugin compilation, please change the ddk path of the file makefile
2. Please change the ddk version in "omg.sh" if necessary.
3. In order to get the NPU model(.om), please run "source env_omg.sh"  in the omg_verify folder, and then "make clean;make" in the plugin folder,  and "bash omg.sh" in the omg_verify folder.
