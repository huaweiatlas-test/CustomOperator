**Interface**

```python
def custom_batch_matmul(
    shape_x, 
    shape_y, 
    dtype, 
    trans_a=False, 
    trans_b=False, 
    kernel_name = "cce_tf_batch_matmul", 
    need_build = False,
    need_print = False):
```

**Description**

Multiplies slices of two tensors in batches(each slice can be viewed as an element of a batch), the output is of the same batch size.

Each of the individual slices can optionally be transposed before multiplication by setting the trans_a or trans_b flag to True, which are by default False. The input tensors are 2-D or higher with the shape [..., r_x, c_x] and [..., r_y, c_y]. The output tensor is 2-D or higher with the shape [..., r_o, c_o], where
r_o = c_x if trans_a else r_x
c_o = r_y if trans_b else c_y

**Args:**

- shape_x : shape of the first tensor x with rank > 1
- shape_y : shape of the second tensor y with the same type and shape with x
- dtype: input tensor's dtype, support:`float16,float32`
- trans_a : if True, shape_x is transposed before multiplication
- trans_b : if True, shape_y is transposed before multiplication
- kernel_name: op's kernel function name
- need_build: whether build CCEC kernel
- need_print: whether print IR

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`

**Notice**

1. Before plugin compilation, please change the ddk path of the file makefile on line 17. 
2. Please change the ddk path of env_te.sh in the omg_verify folder before OMG.
3. In order to get the NPU model(.om), please run "source env_te.sh"  in the omg_verify folder, and then "make clean;make" in the plugin folder,  and "bash omg.sh" in the omg_verify folder.

