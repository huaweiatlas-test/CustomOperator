**Interface**

```python
def custom_Reduction(
    shape, 
    dtype, 
    axis, 
    op, 
    coeff,
    kernel_name="cce_reductionLayer",
    need_build=False, 
    need_print=False):
```

**Description**

Reduce the input tensor on a certain axis, and scale output with coeff.

Suppose we have an n-axis bottom Blob with shape:

(d0, d1, d2, ..., d(m-1), dm, d(m+1), ..., d(n-1))

If axis == m, the output Blob will have shape

(d0, d1, d2, ..., d(m-1))

And the ReductionOp operation is performed (d0 * d1 * d2 * ... * d(m-1)) times, each including (dm * d(m+1) * ... * d(n-1)) individual data.

If axis == 0 (the default), the output Blob always has the empty shape (count 1), performing reduction across the entire input. Often useful for creating new loss functions.

**Args:**

- shape : input tensor's shape
- dtype : input tensor's dtype, support:`float16,float32`
- axis : the first axis to reduce
- op : can only be one of "SUM, ASUM (sum of abs), SUMSQ (sum of sqr), MEAN"
- coeff : scale for output
- kernel_name: op's kernel func name, optional
- need_build: whether build CCEC kernel, default is `False`, optional
- need_print: whether print IR, default is `False`, optional

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`

**Notice**

1. Before plugin compilation, please change the ddk path of the file Makefile on line 12. 
2. Please change the ddk path of env_omg.sh and omg.sh in the omg_verify folder before OMG.
3. In order to get the NPU model(.om), please run "source env_omg.sh" in the omg_verify folder, and then "make clean;make" in the plugin folder, and "bash omg.sh" in the omg_verify folder.