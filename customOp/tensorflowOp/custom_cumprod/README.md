**Interface**

```python
def custom_cumprod(
    shape, 
    dtype, 
    axis=0, 
    exclusive=False, 
    reverse=False, 
    kernel_name="cce_tf_cumprod", 
    need_build=False, 
    need_print=False):
```

**Description**

By default, this op performs an inclusive cumprod, which means that the first element of the input is identical to the first element of the output :

```python
custom_cumprod([a, b, c])  # [a, a * b, a * b * c]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumprod is performed instead:

```python
custom_cumprod([a, b, c], exclusive=True)  # [1, a, a * b]
```

By setting the `reverse` kwarg to `True`, the cumprod is performed in the opposite direction:

```python
custom_cumprod([a, b, c], reverse=True)  # [a * b * c, b * c, c]
```

The `reverse` and `exclusive` kwargs can also be combined:

```python
custom_cumprod([a, b, c], exclusive=True, reverse=True)  # [b * c, c, 1]
```

**Args:**

- shape : shape of the input tensor
- dtype : input tensor's dtype, support:`float16,float32`
- axis : a Tensor of type int32 (default: 0). Must be in the range [-rank(x), rank(x)).
- exclusive : if True, perform exclusive cumprod
- reverse: if True, perform cumprod in the opposite direction
- kernel_name: op's kernel function name
- need_build: whether build CCEC kernel
- need_print: whether print IR

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`

**Notice**

1. Before plugin compilation, please change the ddk path of the file makefile on line 17. 
2. Please change the ddk path of env_te.sh in the omg_verify folder before OMG.
3. In order to get the NPU model(.om), please run "source env_te.sh"  in the omg_verify folder, and then "make clean;make" in the plugin folder,  and "bash omg.sh" in the omg_verify folder.