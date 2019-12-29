    /* Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
    *
    * This program is free software; you can redistribute it and/or modify
    * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the 
    * License.
    *
    * This program is distributed in the hope that it will be useful,
    * but WITHOUT ANY WARRANTY; without even the implied warranty of
    * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    * Apache License for more details at
    * http:// www.apache.org/licenses/LICENSE-2.0
    */
#include <Python.h>
#include "custom/custom_op.h"
#include "framework/omg/register.h"
#include "framework/omg/omg_types.h"
#include "proto/caffe/caffe.pb.h"
#include "operator.h"
#include "attr_value.h"
#include <memory>
#include <string>
#include <vector>
using namespace ge;

namespace domi {
const int DOMI_COMMON_ONE = 1;
const int DOMI_COMMON_TWO = 2;
const int DOMI_COMMON_THREE = 3;

// Caffe ParseParams function
Status CaffecustomExpParseParams(const Message *op_origin, ge::Operator &opDest)
{
    // trans op_origin to layer
    const caffe::LayerParameter *layer = dynamic_cast<const caffe::LayerParameter *>(op_origin);

    // #### Verify the validity of input operator parameters.
    if (nullptr == layer) {
        printf("Dynamic cast op_src to LayerParameter failed\n");
        return FAILED;
    }
    // #### Obtains operator parameters.
    const caffe::custom_ExpParameter &param = layer->custom_exp_param();
    opDest.SetAttr("shift", AttrValue::CreateFrom<AttrValue::FLOAT>(param.shift()));
    opDest.SetAttr("scale", AttrValue::CreateFrom<AttrValue::FLOAT>(param.scale()));
    opDest.SetAttr("base", AttrValue::CreateFrom<AttrValue::FLOAT>(param.base()));

    return SUCCESS;
}

// #### Obtains the processing function of the output tensor description.
Status CaffecustomExpInferShapeAndType(const ge::Operator &op, vector<ge::TensorDesc> &vOutputDesc)
{
    auto tensorDesc = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();
    float shift = 0.0;
    ge::AttrValue shiftAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("shift", shiftAttrValue))
        || (ge::GRAPH_SUCCESS != shiftAttrValue.GetValue<ge::AttrValue::FLOAT>(shift))) {
        printf("GetOpAttr shift  failed!\n ");
    }
    float scale = 1.0;
    ge::AttrValue scaleAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("scale", scaleAttrValue))
        || (ge::GRAPH_SUCCESS != scaleAttrValue.GetValue<ge::AttrValue::FLOAT>(scale))) {
        printf("GetOpAttr scale  failed!\n ");
    }
    float base = -1.0;
    ge::AttrValue baseAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("base", baseAttrValue))
        || (ge::GRAPH_SUCCESS != baseAttrValue.GetValue<ge::AttrValue::FLOAT>(base))) {
        printf("GetOpAttr base  failed!\n ");
    }

    // GetoutputDescription. Notice that don't delete this sentence!
    tensorDesc.SetShape(shape);
    vOutputDesc.push_back(tensorDesc);

    return SUCCESS;
}

// build Te Binary file
Status CaffecustomExpBuildTeBin(const ge::Operator &op, TEBinInfo &teBinInfo)
{
    // ### Parse the parameters.
    float shift = 0.0;
    ge::AttrValue shiftAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("shift", shiftAttrValue))
        || (ge::GRAPH_SUCCESS != shiftAttrValue.GetValue<ge::AttrValue::FLOAT>(shift))) {
        printf("GetOpAttr shift  failed!\n ");
    }
    float scale = 1.0;
    ge::AttrValue scaleAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("scale", scaleAttrValue))
        || (ge::GRAPH_SUCCESS != scaleAttrValue.GetValue<ge::AttrValue::FLOAT>(scale))) {
        printf("GetOpAttr scale  failed!\n ");
    }
    float base = -1.0;
    ge::AttrValue baseAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("base", baseAttrValue))
        || (ge::GRAPH_SUCCESS != baseAttrValue.GetValue<ge::AttrValue::FLOAT>(base))) {
        printf("GetOpAttr base  failed!\n ");
    }

    // ### Parse input tensor description
    ge::TensorDesc input1Desc = op.GetInputDesc(0);

    // ### Parse the input shape value and check whether the value is 4.
    if (input1Desc.GetShape().GetDimNum() != 4) {
        printf("The shape size is %d, which is not 4!\n", (uint32_t)input1Desc.GetShape().GetDimNum());
        return FAILED;
    }

    std::string filePath = "../operator/custom_Exp";
    std::string funcName = "custom_Exp";
    std::string kernelName = "custom_Exp_" + std::to_string(input1Desc.GetShape().GetDim(0)) 
                             + "_" + std::to_string(input1Desc.GetShape().GetDim(DOMI_COMMON_ONE)) 
                             + "_" + std::to_string(input1Desc.GetShape().GetDim(DOMI_COMMON_TWO)) 
                             + "_" + std::to_string(input1Desc.GetShape().GetDim(DOMI_COMMON_THREE));

    // get real path of Py module
    char *cwd = getcwd(NULL, 0);
    if (cwd == NULL) {
        printf("Get current directory path failed!\n");
        return FAILED;
    }
    std::string cwdS(cwd);
    char *realPath = realpath((cwdS + "/" + filePath + ".py").c_str(), NULL);
    if (realPath == NULL) {
        printf("Get real path of Py module failed!\n");
        return FAILED;
    }
    std::string realFilePathString = std::string(realPath);
    std::string realFilePath = realFilePathString.substr(0, realFilePathString.rfind("."));

    std::map<ge::DataType, std::string> operation_map = {
        { ge::DT_UNDEFINED,      "undefined" },
        { ge::DT_FLOAT,          "float32" },
        { ge::DT_FLOAT16,        "float16" },
        { ge::DT_INT8,           "int8" },
        { ge::DT_INT16,          "int16" },
        { ge::DT_INT32,          "int32" },
        { ge::DT_INT64,          "int64" },
        { ge::DT_UINT8,          "uint8" },
        { ge::DT_UINT16,         "uint16" },
        { ge::DT_UINT32,         "uint32" },
        { ge::DT_UINT64,         "uint64" },
        { ge::DT_BOOL,           "bool" },
        { ge::DT_DOUBLE,         "double" },
        { ge::DT_DUAL,           "dual" },
        { ge::DT_DUAL_SUB_INT8,  "dual_sub_int8" },
        { ge::DT_DUAL_SUB_UINT8, "dual_sub_uint8" }
    };

    std::string dtype = operation_map[op.GetInputDesc(0).GetDataType()];

    // i => int; s => string; f => dobule; O => bool, and bool value is Py_True or Py_False.
    te::BuildTeCustomOp(teBinInfo.ddk_version, op.GetName(), realFilePath, funcName,
                        "(i,i,i,i),s, f, f, f, s,O",
                        input1Desc.GetShape().GetDim(0),
                        input1Desc.GetShape().GetDim(DOMI_COMMON_ONE),
                        input1Desc.GetShape().GetDim(DOMI_COMMON_TWO),
                        input1Desc.GetShape().GetDim(DOMI_COMMON_THREE),
                        dtype.c_str(), base,
                        scale, shift,
                        kernelName.c_str(),
                        Py_True);

    // set te op json to teBinInfo
    teBinInfo.bin_file_path = "./kernel_meta/" + kernelName + ".o";
    teBinInfo.json_file_path = "./kernel_meta/" + kernelName + ".json";

    return SUCCESS;
}

REGISTER_CUSTOM_OP("custom_exp_param")                      // test_custom_Exp is the type name of the operator in the OM model. It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
.FrameworkType(CAFFE)                                   // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
.OriginOpType("custom_Exp")                           // custom_Exp indicates the type name of the operator in the caffe framework.
.ParseParamsFn(CaffecustomExpParseParams)              // Op parameters parse function
.InferShapeAndTypeFn(CaffecustomExpInferShapeAndType)  // Set output description and datatype function
.TEBinBuildFn(CaffecustomExpBuildTeBin)                // Build Te op binary function
.ImplyType(ImplyType::TVM);

}  // namespace domi
