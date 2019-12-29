**Interface**

```python
def custom_Exp(
    shape, 
    dtype, 
    gamma, 
    alpha, 
    beta, 
    kernel_name="cce_exp",
    need_build=False,
    need_print=False):
```

**Description**

Computes $y=\gamma^{\alpha x+\beta}$, as specified by the alpha $\alpha$, beta $\beta$, and gamma $\gamma$.

**Args:**

- shape: input tensor's shape
- dtype: input tensor's dtype, support:`float16,float32`
- gamma : $\gamma$ in $y=\gamma^{\alpha x+\beta}$
- alpha : $\alpha$ in $y=\gamma^{\alpha x+\beta}$
- beta : $\beta$ in $y=\gamma^{\alpha x+\beta}$
- kernel_name: op's kernel func name, optional
- need_build: whether build CCEC kernel, default is `False`, optional
- need_print: whether print IR, default is `False`, optional

**Returns:**

No returns, generate op's .o file and .json file(describe op's platform) in `./kernel_meta`

**Notice**

1. Before plugin compilation, please change the ddk path of the file makefile
2. Please change the ddk version in "omg.sh" if necessary.
3. In order to get the NPU model(.om), please run "source env_omg.sh"  in the omg_verify folder, and then "make clean;make" in the plugin folder,  and "bash omg.sh" in the omg_verify folder.
