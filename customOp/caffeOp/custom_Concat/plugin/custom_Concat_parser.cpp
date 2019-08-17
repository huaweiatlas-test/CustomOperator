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
Status Caffecustom_ConcatParseParams(const Message* op_origin, ge::Operator& op_dest)
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
    const caffe::custom_ConcatParameter& param = layer->custom_concat_param();

	op_dest.SetAttr("axis", AttrValue::CreateFrom<AttrValue::INT>(param.axis()));  
    return SUCCESS;
}

// #### Obtains the processing function of the output tensor description. 
Status Caffecustom_ConcatInferShapeAndType(const ge::Operator& op, vector<ge::TensorDesc>& v_output_desc)
{
    auto tensorDesc      = op.GetInputDesc(0);
    auto shape = tensorDesc.GetShape();
    uint32_t axis =  1;
    ge::AttrValue axisAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("axis", axisAttrValue))
        || (ge::GRAPH_SUCCESS != axisAttrValue.GetValue<ge::AttrValue::INT>(axis)))
    {
        printf("GetOpAttr axis  failed!\n ");
    }

    int64_t tensor_num = 0;
    for (size_t i = 0; op.GetInputDesc(i).GetShape().GetShapeSize(); i++)
    {
        tensor_num++;
    }
    
    int64_t dim_at_axis = 0;
    for(size_t i = 0; i < tensor_num; i++)
        dim_at_axis += op.GetInputDesc(i).GetShape().GetDim(i);

	shape.SetDim(axis, dim_at_axis);
    tensorDesc.SetShape(shape);
    v_output_desc.push_back(tensorDesc);
    return SUCCESS;
}


// build Te Binary file
Status Caffecustom_ConcatBuildTeBin(const ge::Operator& op, TEBinInfo& te_bin_info)
{
    Py_Initialize();
    
    std::string FilePath   = "";
	std::string RealFilePath    = "";
    std::string FuncName   = "";
    std::string KernelName = "";
    // ### Parse the parameters. 

    uint32_t axis =  1;
    ge::AttrValue axisAttrValue;
    if ((ge::GRAPH_SUCCESS != op.GetAttr("axis", axisAttrValue))
        || (ge::GRAPH_SUCCESS != axisAttrValue.GetValue<ge::AttrValue::INT>(axis)))
    {
        printf("GetOpAttr axis  failed!\n ");
    }

    // ### Parse input tensor description 
     ge::TensorDesc input1_desc = op.GetInputDesc(0);

    // ### Parse the input shape value and check whether the value is 4.
    if(input1_desc.GetShape().GetDimNum() != 4)
    {
        printf("The shape size is %d, which is not 4!\n", (uint32_t)input1_desc.GetShape().GetDimNum());
        return FAILED;
    }

    FilePath   = "../operator/custom_Concat";
    FuncName   = "custom_Concat";
    KernelName = "custom_Concat_" + std::to_string(input1_desc.GetShape().GetDim(0)) + "_" +  std::to_string(input1_desc.GetShape().GetDim(1)) + "_" +
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
    
    int64_t tensor_num = 0;
    for (size_t i = 0; op.GetInputDesc(i).GetShape().GetShapeSize(); i++)
    {
        tensor_num++;
    }

    PyObject *shapes = PyTuple_New(tensor_num); 
    
     for(size_t k = 0; k < tensor_num; k++)
    {
        PyObject *shape = PyTuple_New(op.GetInputDesc(k).GetShape().GetDimNum());
        for(size_t j = 0; j < op.GetInputDesc(k).GetShape().GetDimNum(); j++)
            PyTuple_SetItem(shape, j, Py_BuildValue("i", op.GetInputDesc(k).GetShape().GetDim(j)));
 
        PyTuple_SetItem(shapes, k, shape);
    }
    
    // i => int; s => string; f => dobule; O => bool, and bool value is Py_True or Py_False

	te::BuildTeCustomOp(te_bin_info.ddk_version, op.GetName(), RealFilePath, FuncName,
		"O, s, i, s, O", 
		shapes,
		dtype.c_str(),
		axis,
		KernelName.c_str(),
		Py_True);

    // set te op json to te_bin_info 
    te_bin_info.bin_file_path  = "./kernel_meta/" + KernelName + ".o";
    te_bin_info.json_file_path = "./kernel_meta/" + KernelName + ".json";
    Py_Finalize();
    return SUCCESS;
}

REGISTER_CUSTOM_OP("custom_concat_param") //test_custom_Concat is the type name of the operator in the OM model. It can be specified randomly and cannot be the same as an existing type name. It is case sensitive. 
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("custom_Concat")  // custom_Concat indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(Caffecustom_ConcatParseParams)  // Op parameters parse function
    .InferShapeAndTypeFn(Caffecustom_ConcatInferShapeAndType)       // Set output description and datatype function
    .TEBinBuildFn(Caffecustom_ConcatBuildTeBin)           // Build Te op binary function
    .ImplyType(ImplyType::TVM);  

}  // namespace domi
