**Interface**

```python
def custom_Tile(
    shape, 
    dtype, 
    tiles, 
    axis=1, 
    kernel_name="cce_caffe_tile_layer",
    need_build=False, 
    need_print=False):
```

**Description**

Copy the input tensor along specified dimensions.

**Args:**

- shape: input tensor's shape
- dtype: input tensor's dtype, support:`float16,float32`
- tiles: the number of copies (tiles) of the tensor to output
- axis: the index of the axis to tile
- kernel_name: op's kernel func name, optional
- need_build: whether build CCEC kernel, default is `False`, optional
- need_print: whether print IR, default is `False`, optional

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`

**Notice**

1. Before plugin compilation, please change the ddk path of the file Makefile on line 12. 
2. Please change the ddk path of env_omg.sh and omg.sh in the omg_verify folder before OMG.
3. In order to get the NPU model(.om), please run "source env_omg.sh" in the omg_verify folder, and then "make clean;make" in the plugin folder, and "bash omg.sh" in the omg_verify folder.