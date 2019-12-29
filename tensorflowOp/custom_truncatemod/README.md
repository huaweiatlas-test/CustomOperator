**Interface**

```python
def custom_truncatemod(
    shape1, 
    shape2, 
    dtype, 
    kernel_name="cce_tf_truncatemod", 
    need_build=False, 
    need_print=False):
```

**Description**

Do element-wise truncatemod operation between two input tensors. 

**Args:**

- shape_x : shape of input data1
- shape_y : shape of input data2
- dtype : input tensor's dtype, support:`float16,float32`
- kernel_name: op's kernel function name
- need_build: whether build CCEC kernel
- need_print: whether print IR

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`

**Notice**

1. Before plugin compilation, please change the ddk path of the file makefile
2. Please change the ddk version in "omg.sh" if necessary.
3. In order to get the NPU model(.om), please run "source env_omg.sh"  in the omg_verify folder, and then "make clean;make" in the plugin folder,  and "bash omg.sh" in the omg_verify folder.

