/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
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
#include "operator.h"
#include "attr_value.h"
#include <memory>
#include <string>
#include <vector>

using namespace ge;
namespace domi
{

// #### Obtains the processing function of the output tensor description. 
Status TFcustom_batch_matmulInferShapeAndType(const ge::Operator& op, vector<ge::TensorDesc>& v_output_desc)
{
    auto tensorDesc      = op.GetInputDesc(0);
	auto tensorDesc_2     = op.GetInputDesc(1); 
    auto shape = tensorDesc.GetShape();
	bool adj_x = false;  
	ge::AttrValue adj_xAttrValue;  
	if ((ge::GRAPH_SUCCESS != op.GetAttr("adj_x", adj_xAttrValue)) || (ge::GRAPH_SUCCESS != adj_xAttrValue.GetValue<AttrValue::BOOL>(adj_x)))  
	{  
	    printf("Get adj_x failed!\n ");  
	}  
	bool adj_y = false;  
	ge::AttrValue adj_yAttrValue;  
	if ((ge::GRAPH_SUCCESS != op.GetAttr("adj_y", adj_yAttrValue)) || (ge::GRAPH_SUCCESS != adj_yAttrValue.GetValue<AttrValue::BOOL>(adj_y)))  
	{  
	    printf("Get adj_y failed!\n ");  
	}  
	
		int64_t shape_x = shape.GetDimNum();
		int64_t shape_y = shape.GetDimNum();
		if (shape_x == 2)
		{
			if (adj_x == false && adj_y == false)
			{
				shape.SetDim(1, tensorDesc_2.GetShape().GetDim(1));
			}
			else if (adj_x == false && adj_y == true)
			{
				shape.SetDim(1, tensorDesc_2.GetShape().GetDim(0));
			}
			else if (adj_x == true && adj_y == false)
			{
				shape.SetDim(0, tensorDesc.GetShape().GetDim(1));
				shape.SetDim(1, tensorDesc_2.GetShape().GetDim(1));
			}
			else
			{
				shape.SetDim(0, tensorDesc.GetShape().GetDim(1));
				shape.SetDim(1, tensorDesc_2.GetShape().GetDim(0));
			}
		}
		else
		{
			if (adj_x == false && adj_y == false)
			{
				shape.SetDim(shape_x-1, tensorDesc_2.GetShape().GetDim(shape_y-1));
			}
			else if (adj_x == false && adj_y == true)
			{
				shape.SetDim(shape_x-1, tensorDesc_2.GetShape().GetDim(shape_y-2));
			}
			else if (adj_x == true && adj_y == false)
			{
				shape.SetDim(shape_x-2, tensorDesc.GetShape().GetDim(shape_x-1));
				shape.SetDim(shape_x-1, tensorDesc_2.GetShape().GetDim(shape_y-1));
			}
			else
			{
				shape.SetDim(shape_x-2, tensorDesc.GetShape().GetDim(shape_x-1));
				shape.SetDim(shape_x-1, tensorDesc_2.GetShape().GetDim(shape_y-2));
			}

		}

    tensorDesc.SetShape(shape);
    v_output_desc.push_back(tensorDesc);

    return SUCCESS;

}


// build Te Binary file
Status TFcustom_batch_matmulBuildTeBin(const ge::Operator& op, TEBinInfo& te_bin_info)
{
    std::string FilePath   = "";
	std::string RealFilePath = "";
    std::string FuncName   = "";
    std::string KernelName = "";
	
    // ### Parses the parameters. 
	bool adj_x = false;  
	ge::AttrValue adj_xAttrValue;  
	if ((ge::GRAPH_SUCCESS != op.GetAttr("adj_x", adj_xAttrValue)) || (ge::GRAPH_SUCCESS != adj_xAttrValue.GetValue<AttrValue::BOOL>(adj_x)))  
	{  
	    printf("Get bool failed!\n ");  
	}  
	bool adj_y = false;  
	ge::AttrValue adj_yAttrValue;  
	if ((ge::GRAPH_SUCCESS != op.GetAttr("adj_y", adj_yAttrValue)) || (ge::GRAPH_SUCCESS != adj_yAttrValue.GetValue<AttrValue::BOOL>(adj_y)))  
	{  
	    printf("Get bool failed!\n ");  
	}  
    
	// ### Parse input tensor description 
	TensorDesc input_desc     = op.GetInputDesc(0); 
	TensorDesc input_desc_2     = op.GetInputDesc(1); 

    FilePath   = "../operator/custom_batch_matmul";
    FuncName   = "custom_batch_matmul";
    KernelName = "custom_batch_matmul" + std::to_string(input_desc.GetShape().GetDim(0)) + std::to_string(input_desc.GetShape().GetDim(1));

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
 		"(i,i,i,i),(i,i,i,i),s, O, O, s,O", 
		input_desc.GetShape().GetDim(0), input_desc.GetShape().GetDim(1), input_desc.GetShape().GetDim(2), input_desc.GetShape().GetDim(3),
		input_desc_2.GetShape().GetDim(0), input_desc_2.GetShape().GetDim(1), input_desc_2.GetShape().GetDim(2), input_desc_2.GetShape().GetDim(3),
		dtype.c_str(),
	    adj_x ? Py_True: Py_False,  
	    adj_y ? Py_True: Py_False,  
		KernelName.c_str(),
        Py_True);

    // set te op json to te_bin_info 
    te_bin_info.bin_file_path  = "./kernel_meta/" + KernelName + ".o";
    te_bin_info.json_file_path = "./kernel_meta/" + KernelName + ".json";
 
    return SUCCESS;
}

REGISTER_CUSTOM_OP("custom_batch_matmul") //custom_batch_matmul is the type name of the operator in the OM model. It can be specified randomly and cannot be the same as an existing type name. It is case sensitive. 
    .FrameworkType(TENSORFLOW)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("BatchMatMul")  // // batch_matmul indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(AutoMappingFn)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .InferShapeAndTypeFn(TFcustom_batch_matmulInferShapeAndType)       // Set output description and datatype function
    .TEBinBuildFn(TFcustom_batch_matmulBuildTeBin) // Build Te op binary function
    .ImplyType(ImplyType::TVM) // Implementation type. Enumerated type, The options are as follows: TVM, AI_CPU.
    .Formats({DOMI_TENSOR_ND},{DOMI_TENSOR_ND});   //  #### Format of the input and output

}  // namespace domi
