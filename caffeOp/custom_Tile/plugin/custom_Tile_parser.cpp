/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the 
License.
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
Status CaffecustomTileParseParams(const Message *opOrigin, ge::Operator &opDest)
{
    // trans opOrigin to layer
    const caffe::LayerParameter *layer = dynamic_cast<const caffe::LayerParameter *>(opOrigin);

    // #### Verify the validity of input operator parameters.
    if (nullptr == layer) {
        printf("Dynamic cast op_src to LayerParameter failed\n");
        return FAILED;
    }
    // #### Obtains operator parameters.
    const caffe::custom_TileParameter &param = layer->custom_tile_param();
    opDest.SetAttr("tiles", AttrValue::CreateFrom<AttrValue::INT>(param.tiles()));
    opDest.SetAttr("axis", AttrValue::CreateFrom<AttrValue::INT>(param.axis()));

    return SUCCESS;
}

// #### Obtains the processing function of the output tensor description.
Status CaffecustomTileInferShapeAndType(const ge::Operator &op, vector<ge::TensorDesc> &vOutputDesc)
{
    auto tensorDesc = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();
    uint32_t tiles = 0;
    ge::AttrValue tilesAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("tiles", tilesAttrValue))
        || (ge::GRAPH_SUCCESS != tilesAttrValue.GetValue<ge::AttrValue::INT>(tiles))) {
        printf("GetOpAttr tiles  failed!\n ");
    }
    uint32_t axis = 1;
    ge::AttrValue axisAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("axis", axisAttrValue))
        || (ge::GRAPH_SUCCESS != axisAttrValue.GetValue<ge::AttrValue::INT>(axis))) {
        printf("GetOpAttr axis  failed!\n ");
    }

    shape.SetDim(axis, shape.GetDim(axis) * tiles);
    tensorDesc.SetShape(shape);
    vOutputDesc.push_back(tensorDesc);

    return SUCCESS;
}

// build Te Binary file
Status CaffecustomTileBuildTeBin(const ge::Operator &op, TEBinInfo &teBinInfo)
{
    // ### Parse the parameters.
    uint32_t tiles = 0;
    ge::AttrValue tilesAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("tiles", tilesAttrValue))
        || (ge::GRAPH_SUCCESS != tilesAttrValue.GetValue<ge::AttrValue::INT>(tiles))) {
        printf("GetOpAttr tiles  failed!\n ");
    }
    uint32_t axis = 1;
    ge::AttrValue axisAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("axis", axisAttrValue))
        || (ge::GRAPH_SUCCESS != axisAttrValue.GetValue<ge::AttrValue::INT>(axis))) {
        printf("GetOpAttr axis  failed!\n ");
    }

    // ### Parse input tensor description
    ge::TensorDesc input1Desc = op.GetInputDesc(0);

    // ### Parse the input shape value and check whether the value is 4.
    if (input1Desc.GetShape().GetDimNum() != 4) {
        printf("The shape size is %d, which is not 4!\n", (uint32_t)input1Desc.GetShape().GetDimNum());
        return FAILED;
    }

    std::string filePath = "../operator/custom_Tile";
    std::string funcName = "custom_Tile";
    std::string kernelName = "custom_Tile_" + std::to_string(input1Desc.GetShape().GetDim(0)) 
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

    // i => int; s => string; f => dobule; O => bool, and bool value is Py_True or Py_False
    te::BuildTeCustomOp(teBinInfo.ddk_version, op.GetName(), realFilePath, funcName,
                        "(i,i,i,i),s, i, i, s,O",
                        input1Desc.GetShape().GetDim(0),
                        input1Desc.GetShape().GetDim(DOMI_COMMON_ONE),
                        input1Desc.GetShape().GetDim(DOMI_COMMON_TWO),
                        input1Desc.GetShape().GetDim(DOMI_COMMON_THREE),
                        dtype.c_str(), axis, tiles,
                        kernelName.c_str(), Py_True);

    // set te op json to teBinInfo
    teBinInfo.bin_file_path = "./kernel_meta/" + kernelName + ".o";
    teBinInfo.json_file_path = "./kernel_meta/" + kernelName + ".json";

    return SUCCESS;
}

REGISTER_CUSTOM_OP("custom_tile_param")                     // test_custom_Tile is the type name of the operator in the OM model. It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
.FrameworkType(CAFFE)                                   // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
.OriginOpType("custom_Tile")                           // custom_Tile indicates the type name of the operator in the caffe framework.
.ParseParamsFn(CaffecustomTileParseParams)              // Op parameters parse function
.InferShapeAndTypeFn(CaffecustomTileInferShapeAndType)  // Set output description and datatype function
.TEBinBuildFn(CaffecustomTileBuildTeBin)                // Build Te op binary function
.ImplyType(ImplyType::TVM);

}  // namespace domi
