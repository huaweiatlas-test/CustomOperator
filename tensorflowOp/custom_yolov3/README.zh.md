[EN](README.md)|CN

# **YOLOv3 D模型DEMO**

本DEMO提供了YOLOv3 D模型推理实现，其中YOLOv3模型由于最后yolo层D框架不支持，因此pb模型去除最后一层，该层逻辑在D框架后处理中实现。



### 模型转换

通过使用OMG工具转换tensorflow的pb模型，得到YOLOv3的D模型，其中如下图所示，需修改配置项Model Image Format选为RGB888_U8，Multiplying Factor的三个数值均填0.0039216，并将Mean Less开关设置为off。

![1570709301706](./assets/1570709301706.png)

### yolo层实现及调用

由于D框架不支持tensorflow中实现的yolo层，所以在pb文件中去除最后一层，并在D上实现，实现代码为MindInferenceEngine_1中的yolo.h。

在模型推理结束后，将三个检测层的输出送至yolo层处理，最终输出所有目标的矩形框、类别标签及置信度。

![1571036915253](./assets/1571036915253.png)

最终输出detectBox为：

![1571036988369](./assets/1571036988369.png)

DetectBox为目标框结构体，x、y为目标框的中心点坐标，width、height为目标框的宽高，class_id为目标的类别标签，prob为目标的置信度。

### D模型测试

推理结果目前输出到ERROR级别的log中，如下图第155行日志所示：

![1571040131345](./assets/1571040131345.png)

最终业务层需要将矩形框信息转换为原始图像中的矩形框信息：

![1571040325401](./assets/1571040325401.png)

原始图像宽高为src_w=500，src_h=421，目标框计算为x2=x&times;src_w, y2=y&times;src_h, width2=width&times;src_w, height2=height&times;src_h。

测试结果如下图所示：

![1571040788039](./assets/1571040788039.png)

## 其他

对于提供的工程，若要导入mindstudio，则需打包成zip后上传整个工程，且相关配置需对应修改，同时需上传输入图片作为自定义的Dataset和模型作为自定义的模型。

对于ImagePreProcess Engine,当前代码需将其resize宽高改为416*416. 即输入需与实际模型匹配