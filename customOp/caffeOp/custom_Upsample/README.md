**Interface**

```python
def custom_Upsample(
    shape, 
    dtype, 
    scale, 
    data_format="channels_last", 
    kernel_name="cce_darknet_upsample",
    need_build=False, need_print=False):
```

**Description**

Upsamples the input tensor, i.e., repeating the rows and columns of the input tensor by `scale`.

**Args:**

- shape: input tensor's shape
- dtype: input tensor's dtype, support:`float16,float32`
- scale: the upsampling factors
- data_format: "channels_last" or "channels_first"
- kernel_name: op's kernel func name, optional
- need_build: whether build CCEC kernel, default is `False`, optional
- need_print: whether print IR, default is `False`, optional

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`

**Notice**

1. Before plugin compilation, please change the ddk path of the file Makefile on line 12. 
2. Please change the ddk path of env_omg.sh and omg.sh in the omg_verify folder before OMG.
3. In order to get the NPU model(.om), please run "source env_omg.sh" in the omg_verify folder, and then "make clean;make" in the plugin folder, and "bash omg.sh" in the omg_verify folder.