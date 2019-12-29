    /* Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
    *
    * This program is free software; you can redistribute it and/or modify
    * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
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
// #### Obtains the processing function of the output tensor description.
Status TFcustomLog1pInferShapeAndType(const ge::Operator &op, vector<ge::TensorDesc> &v_output_desc)
{
    auto tensorDesc = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();

    tensorDesc.SetShape(shape);
    v_output_desc.push_back(tensorDesc);

    return SUCCESS;
}

// build Te Binary file
Status TFcustomLog1pBuildTeBin(const ge::Operator &op, TEBinInfo &te_bin_info)
{
    // ### Parse input tensor description
    TensorDesc inputDesc = op.GetInputDesc(0);

    std::string filePath = "../operator/custom_log1p";
    std::string funcName = "custom_log1p";
    std::string kernelName = "custom_log1p" + std::to_string(inputDesc.GetShape().GetDim(0)) + std::to_string(inputDesc.GetShape().GetDim(DOMI_COMMON_ONE));

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

    // i => int; s => string; f => dobule; O => bool, and bool value is Py_True or Py_False
    te::BuildTeCustomOp(te_bin_info.ddk_version, op.GetName(), realFilePath, funcName,
                        "(i,i,i,i),s, s,O",
                        inputDesc.GetShape().GetDim(0), inputDesc.GetShape().GetDim(DOMI_COMMON_ONE),
                        inputDesc.GetShape().GetDim(DOMI_COMMON_TWO), inputDesc.GetShape().GetDim(DOMI_COMMON_THREE),
                        dtype.c_str(),
                        kernelName.c_str(),
                        Py_True);

    // set te op json to te_bin_info
    te_bin_info.bin_file_path = "./kernel_meta/" + kernelName + ".o";
    te_bin_info.json_file_path = "./kernel_meta/" + kernelName + ".json";

    return SUCCESS;
}

REGISTER_CUSTOM_OP("custom_log1p")                        // custom_log1p is the type name of the operator in the OM model. It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
.FrameworkType(TENSORFLOW)                            // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
.OriginOpType("Log1p")                         // Log1p indicates the type name of the operator in the caffe framework.
.ParseParamsFn(AutoMappingFn)                         // AutoMappingFn indicates automatic mapping the parameters of op.
.InferShapeAndTypeFn(TFcustomLog1pInferShapeAndType)  // Set output description and datatype function
.TEBinBuildFn(TFcustomLog1pBuildTeBin)                // Build Te op binary function
.ImplyType(ImplyType::TVM)                            // Implementation type. Enumerated type, The options are as follows: TVM, AI_CPU.
.Formats({ DOMI_TENSOR_ND }, { DOMI_TENSOR_ND });     // #### Format of the input and output

}  // namespace domi
