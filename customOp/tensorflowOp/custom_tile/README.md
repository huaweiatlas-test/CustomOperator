**Interface**

```python
def custom_tile(
    shape, 
    multiples, 
    dtype, 
    kernel_name="cce_tile", 
    need_build=False,
    need_print=False):
```

**Description**

This operation creates a new tensor by replicating `input` `multiples` times. The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements, and the values of `input` are replicated `multiples[i]` times along the 'i'th dimension. For example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.

**Args:**

- shape : shape of the input tensor
- multiples : shape of multiples
- dtype : input tensor's dtype, support:`float16,float32`
- kernel_name: op's kernel function name
- need_build: whether build CCEC kernel
- need_print: whether print IR

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`

**Notice**

1. Before plugin compilation, please change the ddk path of the file makefile on line 17. 
2. Please change the ddk path of env_te.sh in the omg_verify folder before OMG.
3. In order to get the NPU model(.om), please run "source env_te.sh"  in the omg_verify folder, and then "make clean;make" in the plugin folder,  and "bash omg.sh" in the omg_verify folder.