EN|[CN](README.zh.md)

## Custom Operator Samples

There are 34 custom operator samples given in this repository.

### Introduction to the TE and TVM

TE (Tensor Engine) is a custom operator development framework based on the [TVM](https://tvm.ai/about) (Tensor Virtual Machine).

TVM is [an open source project of the Github](https://github.com/dmlc/tvm). It aims to further abstract the generation rules of operators, divide the operators into operation primitives, and combine the operators when necessary. According to the definition of the calculation process of operators, the TVM uses the Schedule technology and the Codegen technology to generate the operator for the specified hardware. For more information about the TVM, see [TVM Tutorials](https://docs.tvm.ai/tutorials/tensor_expr_get_started.html) and [TVM Forum] (https://discuss.tvm.ai/).

The TE is the repackage of the TVM. The functions in the TE are classified into two types: [TVM primitive] (https://docs.tvm.ai/api/python/tvm.html) and TE-DSL.

In this repository, some operators use the TVM primitive, and some operators use the TE-DSL primitive.

### Supported Product

Atlas 300 (Model 3000), Atlas 300 (Model 3010)

### Supported Version

1.3.2.B893

1.31.T15.B150

Supported versions are obtained by the following command:

```bash
npu-smi info
```

### Directory Structure

The directory structure is as follows.

```bash
.
├── customOp
│   ├── caffeOp
│   │   ├── custom_Concat
│   │   ├── custom_Exp
│   │   ├── custom_Log
│   │   ├── custom_Power
│   │   ├── custom_Reduction
│   │   ├── custom_Tile
│   │   ├── custom_Upsample
│   │   └── SpatialTransformer
│   └── tensorflowOp
│       ├── custom_abs
│       ├── custom_batch_matmul
│       ├── custom_ceil
│       ├── custom_equal
│       ├── custom_exp
│       ├── custom_expm1
│       ├── custom_floor
│       ├── custom_greater_equal
│       ├── custom_l2_loss
│       ├── custom_log
│       ├── custom_log1p
│       ├── custom_logical_and
│       ├── custom_logical_not
│       ├── custom_maximum
│       ├── custom_minimum
│       ├── custom_mod
│       ├── custom_pow
│       ├── custom_rint
│       ├── custom_round
│       ├── custom_sign
│       ├── custom_sqrt
│       ├── custom_square
│       ├── custom_squared_difference
│       ├── custom_subtract
│       └── custom_truncatemod
|       |__ custom_yolov3
└── README.md
```

In the preceding information, README.md is the main text.

customOp stores the custom operator, which consists of the TensorFlow operator and Caffe operator. In all custom operators, the SpatialTransformer is developed by using the MindSpore Studio, and other operators are developed by using command lines. (Therefore, the directory structure of the SpatialTransformer is different from that of other operators.) In addition, custom_yolov3 is developed using the Graph engine rather than the TE method.

- Caffe custom operator directory, that is, the structure of ./caffeOp/custom_xxx is as follows.

  ```bash
  .
  ├── omg_verify
  │   ├── custom_xxx.caffemodel
  │   ├── custom_xxx.prototxt
  │   ├── env_omg.sh
  │   └── omg.sh
  ├── operator
  │   └── custom_xxx.py
  ├── plugin
  │   ├── custom_xxx_parser.cpp
  │   ├── Makefile
  │   └── proto
  │       └── caffe
  │           └── caffe.proto
  └── README.md
  ```

  The custom_xxx.py file in the operator directory is the operator code file.

  The custom_xxx_parser.cpp file in the plugin directory is the plug-in code file. The caffe.proto file contains the message field of the custom operator.

  The omg_verify directory stores the Caffe network (including custom_xxx.caffemodel and custom_xxx.prototxt) that contains the custom operator and the OMG (model conversion) script (env_omg.sh and omg.sh) in command line mode.

- TensorFlow custom operator directory, that is, the structure of ./tensorflowOp/custom_xxx is as follows.

  ```bash
  .
  ├── omg_verify
  │   ├── custom_xxx.pb
  │   ├── env_te.sh
  │   └── omg.sh
  ├── operator
  │   └── custom_xxx.py
  ├── plugin
  │   ├── custom_xxx_tf_parser.cpp
  │   └── Makefile
  └── README.md
  ```

  The custom_xxx.py file in the operator directory is the operator code file.

  The custom_xxx_tf_parser.cpp file in the plugin directory is the plug-in code file.

  The omg_verify directory stores the TensorFlow network (custom_xxx.pb) that contains the custom operator and the OMG (model conversion) script (env_te.sh and omg.sh) in command line mode.

### Plug-In Compilation

- Caffe Operator:

  1) Modify the DDK path in the makefile file of the custom_xxx/plugin/ directory.

  2) Execute the source env_omg.sh file to set environment variables.

  3) Run make clean;make in the custom_xxx/plugin/ directory.

- TensorFlow Operator:

  1) Modify the DDK path in the makefile file of the custom_xxx/plugin/ directory.

  2) Execute the source env_omg.sh file to set environment variables.

  3) Run make clean;make in the custom_xxx/plugin/ directory.

### OMG (Model Conversion)

Based on the plug-in compilation, the following describes how to perform the OMG (model conversion) in command line mode.

- Caffe Operator:

  1) If the new DDK, is used, modify the ddk_version parameter of the omg.sh file. If the DDK runs on the Atlas 300 (Model 3000) or CentOS, the OMG path also needs to be modified.

  2) Execute the source env_omg.sh and bash omg.sh files in sequence in the custom_xxx/omg_verify/ directory.

- TensorFlow Operator:

  1) Modify the value of ddk_version in the omg.sh file to the actual DDK version. If the DDK runs on the Atlas 300 (Model 3000) or CentOS, the OMG path also needs to be modified.

  2) Execute the source env_omg.sh and bash omg.sh files in sequence in the custom_xxx/omg_verify/ directory.
