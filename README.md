## 自定义算子样例

本仓库给出了36个 TE 自定义算子的样例。

### TE 与 TVM 简介

[TE](https://www.huawei.com/minisite/ascend/cn/filedetail_20.html) （Tensor Engine）是基于 [TVM](https://tvm.ai/about) （Tensor Virtual Machine）的自定义算子开发框架。

TVM 是 [Github 的开源项目](https://github.com/dmlc/tvm)， 旨在将各算子的生成规则进一步抽象，将算子本身分成各个操作原语，在需要的时候加以组合。TVM 会根据算子的计算过程的定义，使用 Schedule 技术和 Codegen 技术，生成对指定硬件的算子。TVM 的相关问题可参考 [TVM Tutorials](https://docs.tvm.ai/tutorials/tensor_expr_get_started.html) 与 [TVM论坛](https://discuss.tvm.ai/)。

TE 是对 TVM 的再次封装，TE 中的函数包括两大类：[TVM原语](https://docs.tvm.ai/api/python/tvm.html) 与 TE-DSL。

在本仓库的算子中，有的采用了 TVM 原语，有的采用了 TE-DSL。

### 文件说明

本仓库的目录结构如下：

```bash
.
├── customOp
│   ├── caffeOp
│   │   ├── custom_Concat
│   │   ├── custom_Exp
│   │   ├── custom_Log
│   │   ├── custom_Power
│   │   ├── custom_Reduction
│   │   ├── custom_Tile
│   │   ├── custom_Upsample
│   │   └── SpatialTransformer
│   └── tensorflowOp
│       ├── custom_abs
│       ├── custom_batch_matmul
│       ├── custom_ceil
│       ├── custom_cumprod
│       ├── custom_equal
│       ├── custom_exp
│       ├── custom_expm1
│       ├── custom_floor
│       ├── custom_greater_equal
│       ├── custom_l2_loss
│       ├── custom_log
│       ├── custom_log1p
│       ├── custom_logical_and
│       ├── custom_logical_not
│       ├── custom_maximum
│       ├── custom_minimum
│       ├── custom_mod
│       ├── custom_negative
│       ├── custom_pow
│       ├── custom_rint
│       ├── custom_round
│       ├── custom_sign
│       ├── custom_sqrt
│       ├── custom_square
│       ├── custom_squared_difference
│       ├── custom_subtract
│       ├── custom_tile
│       └── custom_truncatemod
└── README.md
```

其中，README.md 为本文。

customOp 存放着自定义算子，它包括两部分：tensorflow 算子与 caffe 算子。在所有自定义算子中，SpatialTransformer 通过 MindSpore Studio 开发，其余通过命令行方式开发（因此 SpatialTransformer 的目录结构与其余算子的目录结构有所不同）。

- caffe自定义算子目录，即 ./caffeOp/custom_xxx 的结构为：

  ```bash
  .
  ├── omg_verify
  │   ├── custom_xxx.caffemodel
  │   ├── custom_xxx.prototxt
  │   ├── env_omg.sh
  │   └── omg.sh
  ├── operator
  │   └── custom_xxx.py
  ├── plugin
  │   ├── custom_xxx_parser.cpp
  │   ├── Makefile
  │   └── proto
  │       └── caffe
  │           └── caffe.proto
  └── README.md
  ```

  其中，operator 目录下的 custom_xxx.py 为算子代码文件。

  plugin 目录下的 custom_xxx_parser.cpp 为插件代码文件，caffe.proto 里包含了该自定义算子的 message 字段。

  omg_verify 目录下存放着包括该自定义算子的 caffe 网络（包括 custom_xxx.caffemodel 与 custom_xxx.prototxt），以及命令行方式的 OMG （模型转换）的脚本（即 env_omg.sh 与 omg.sh）。

- tensorflow 自定义算子目录，即 ./tensorflowOp/custom_xxx 的结构为：

  ```bash
  .
  ├── omg_verify
  │   ├── custom_xxx.pb
  │   ├── env_te.sh
  │   └── omg.sh
  ├── operator
  │   └── custom_xxx.py
  ├── plugin
  │   ├── custom_xxx_tf_parser.cpp
  │   └── Makefile
  └── README.md
  ```

  其中，operator 目录下的 custom_xxx.py 为算子代码文件。

  plugin 目录下的 custom_xxx_tf_parser.cpp 为插件代码文件。

  omg_verify 目录下存放着包括该自定义算子的 tensorflow 网络（即 custom_xxx.pb），以及命令行方式的 OMG（模型转换）的脚本（即 env_te.sh 与 omg.sh）。

### 插件编译

- 对于 caffe 算子：

  1）修改 custom_xxx/plugin/ 的 Makefile 文件中的 DDK 路径（位于第12行）。

  2）修改 custom_xxx/omg_verify/env_omg.sh 的 DDK_PATH，并执行 source env_omg.sh。

  3）在 custom_xxx/plugin/ 目录下执行 make clean; make。

- 对于 tensorflow 算子：

  1）修改 custom_xxx/plugin/ 的 Makefile 文件中的 DDK 路径（位于第17行）。

  2）修改 custom_xxx/omg_verify/env_te.sh 的 DDK_PATH，并执行 source env_te.sh。

  3）在 custom_xxx/plugin/ 目录下执行 make clean; make。

### OMG（模型转换）

下面介绍通过命令行方式进行 OMG（模型转换）。

- 对于 caffe 算子：

  1）修改 custom_xxx/omg_verify/env_omg.sh 的 DDK_PATH。

  2）修改 omg.sh 的 DDK_PATH 与 ddk_version 参数。

  3）在 custom_xxx/omg_verify/ 目录下依次执行 source env_omg.sh 与 bash omg.sh。

- 对于 tensorflow 算子：

  1）修改 custom_xxx/omg_verify/env_omg.sh 的 DDK_PATH。

  2）修改 omg.sh 的 ddk_version 参数为实际的 DDK 版本。
  
  2）在 custom_xxx/omg_verify/ 目录下依次执行 source env_te.sh 与 bash omg.sh。