**Interface**

```python
def custom_Concat(
    shapes, 
    dtype, 
    axis, 
    kernel_name="concat", 
    need_build=False, 
    need_print=False):
```

**Description**

Concatenates the list of tensors `values` along dimension `axis`. If `values[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, the concatenated result has shape:

$[D0, D1, ... \color{Blue}{Daxis}$, $...Dn]$

where

$Raxis = sum(Daxis(i))$

That is, the data from the input tensors is joined along the `axis` dimension.

The number of dimensions of the input tensors must match, and all dimensions except `axis` must be equal.

**Args:**

- shape: input tensor's shape
- dtype: input tensor's dtype, support:`float16,float32`
- axis : dimension along which to concatenate
- kernel_name: op's kernel func name, optional
- need_build: whether build CCEC kernel, default is `False`, optional
- need_print: whether print IR, default is `False`, optional

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`

**Notice**

1. Before plugin compilation, please change the ddk path of the file makefile
2. Please change the ddk version in "omg.sh" if necessary.
3. In order to get the NPU model(.om), please run "source env_omg.sh"  in the omg_verify folder, and then "make clean;make" in the plugin folder,  and "bash omg.sh" in the omg_verify folder.
