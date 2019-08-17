/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the 
License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
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

namespace domi
{
// Caffe ParseParams function
Status Caffecustom_PowerParseParams(const Message* op_origin, ge::Operator& op_dest)
{
    // trans op_origin to layer
    const caffe::LayerParameter* layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);


    // #### Verify the validity of input operator parameters.
    if (nullptr == layer)
    {
        printf("Dynamic cast op_src to LayerParameter failed\n");
        return FAILED;
    }
    // #### Obtains operator parameters.
    const caffe::custom_PowerParameter& param = layer->custom_power_param();
	
	op_dest.SetAttr("shift", AttrValue::CreateFrom<AttrValue::FLOAT>(param.shift()));  
	op_dest.SetAttr("scale", AttrValue::CreateFrom<AttrValue::FLOAT>(param.scale()));  
	op_dest.SetAttr("power", AttrValue::CreateFrom<AttrValue::FLOAT>(param.power()));  
    
    return SUCCESS;
}

// #### Obtains the processing function of the output tensor description. 
Status Caffecustom_PowerInferShapeAndType(const ge::Operator& op, vector<ge::TensorDesc>& v_output_desc)
{
    auto tensorDesc      = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();
    tensorDesc.SetShape(shape);
    v_output_desc.push_back(tensorDesc);
    return SUCCESS;

}


// build Te Binary file
Status Caffecustom_PowerBuildTeBin(const ge::Operator& op, TEBinInfo& te_bin_info)
{
    std::string FilePath   = "";
	std::string RealFilePath    = "";
    std::string FuncName   = "";
    std::string KernelName = "";
    // ### Parse the parameters. 
    float shift =  0.0;
    ge::AttrValue shiftAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("shift", shiftAttrValue))
        || (ge::GRAPH_SUCCESS != shiftAttrValue.GetValue<ge::AttrValue::FLOAT>(shift)))
    {
        printf("GetOpAttr shift  failed!\n ");
    }
    float scale =  1.0;
    ge::AttrValue scaleAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("scale", scaleAttrValue))
        || (ge::GRAPH_SUCCESS != scaleAttrValue.GetValue<ge::AttrValue::FLOAT>(scale)))
    {
        printf("GetOpAttr scale  failed!\n ");
    }
    float power =  1.0;
    ge::AttrValue powerAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("power", powerAttrValue))
        || (ge::GRAPH_SUCCESS != powerAttrValue.GetValue<ge::AttrValue::FLOAT>(power)))
    {
        printf("GetOpAttr power  failed!\n ");
    }

    // ### Parse input tensor description 
     ge::TensorDesc input1_desc = op.GetInputDesc(0);

    // ### Parse the input shape value and check whether the value is 4.
    if(input1_desc.GetShape().GetDimNum() != 4)
    {
        printf("The shape size is %d, which is not 4!\n", (uint32_t)input1_desc.GetShape().GetDimNum());
        return FAILED;
    }

    FilePath   = "../operator/custom_Power";
    FuncName   = "custom_Power";
    KernelName = "custom_Power_" + std::to_string(input1_desc.GetShape().GetDim(0)) + "_" +  std::to_string(input1_desc.GetShape().GetDim(1)) + "_" +
         std::to_string(input1_desc.GetShape().GetDim(2)) + "_" + std::to_string(input1_desc.GetShape().GetDim(3));

    //get real path of Py module
	char *cwd = getcwd(NULL, 0);
	if (cwd == NULL) {
		printf("Get current directory path failed!\n");
		return FAILED;
	}
	std::string cwd_s(cwd);
	char *real_path = realpath((cwd_s + "/" + FilePath + ".py").c_str(), NULL);
	if (real_path == NULL) {
		printf("Get real path of Py module failed!\n");
		return FAILED;
	}
	std::string RealFilePath_ = std::string(real_path);
	RealFilePath = RealFilePath_.substr(0, RealFilePath_.rfind("."));

	std::map<ge::DataType, std::string> operation_map = {
		{ ge::DT_UNDEFINED, "undefined" },
		{ ge::DT_FLOAT, "float32" },
		{ ge::DT_FLOAT16, "float16" },
		{ ge::DT_INT8, "int8" },
		{ ge::DT_INT16, "int16" },
		{ ge::DT_INT32, "int32" },
		{ ge::DT_INT64, "int64" },
		{ ge::DT_UINT8, "uint8" },
		{ ge::DT_UINT16, "uint16" },
		{ ge::DT_UINT32, "uint32" },
		{ ge::DT_UINT64, "uint64" },
        { ge::DT_BOOL, "bool" },
		{ ge::DT_DOUBLE, "double" },
		{ ge::DT_DUAL, "dual" },
		{ ge::DT_DUAL_SUB_INT8, "dual_sub_int8" },
		{ ge::DT_DUAL_SUB_UINT8, "dual_sub_uint8" }
	};

    std::string dtype = operation_map[op.GetInputDesc(0).GetDataType()];

    // i => int; s => string; f => dobule; O => bool, and bool value is Py_True or Py_False

    te::BuildTeCustomOp(te_bin_info.ddk_version, op.GetName(), RealFilePath, FuncName,
        "(i,i,i,i),s, f, f, f, s,O", 
        input1_desc.GetShape().GetDim(0), input1_desc.GetShape().GetDim(1), input1_desc.GetShape().GetDim(2), input1_desc.GetShape().GetDim(3),
        dtype.c_str(),
        power,
        scale,
        shift,
        KernelName.c_str(),
        Py_True);

    // set te op json to te_bin_info 
    te_bin_info.bin_file_path  = "./kernel_meta/" + KernelName + ".o";
    te_bin_info.json_file_path = "./kernel_meta/" + KernelName + ".json";

    return SUCCESS;
}

REGISTER_CUSTOM_OP("custom_power_param") //test_custom_Power is the type name of the operator in the OM model. It can be specified randomly and cannot be the same as an existing type name. It is case sensitive. 
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("custom_Power")  // custom_Power indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(Caffecustom_PowerParseParams)  // Op parameters parse function
    .InferShapeAndTypeFn(Caffecustom_PowerInferShapeAndType)       // Set output description and datatype function
    .TEBinBuildFn(Caffecustom_PowerBuildTeBin)           // Build Te op binary function
    .ImplyType(ImplyType::TVM);  

}  // namespace domi
