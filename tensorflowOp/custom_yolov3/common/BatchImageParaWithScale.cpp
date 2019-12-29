/**
* @file BatchImageParaWithScale.cpp
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-4-25
*/
#include <iostream>
#include "BatchImageParaWithScale.h"
#include<stdio.h>
#include<dirent.h>
#include<regex>
#include <fcntl.h>

template <class Archive>
void serialize(Archive& arParam, ScaleInfoT& dataParam)
{
    arParam(dataParam.scale_width, dataParam.scale_height);
}

template <class Archive>
void serialize(Archive& arParam, ResizeInfo& dataParam)
{
    arParam(dataParam.resize_width, dataParam.resize_height);
}

template <class Archive>
void serialize(Archive& arParam, CropInfo& dataParam)
{
    arParam(dataParam.point_x, dataParam.point_y, dataParam.crop_width, dataParam.crop_height);
}

template <class Archive>
void serialize(Archive& arParam, NewImageParaT& dataParam)
{
    arParam(dataParam.f_info, dataParam.img, dataParam.scale_info, dataParam.resize_info, dataParam.crop_info);
}

template <class Archive>
void serialize(Archive& arParam, NewImageParaT2& dataParam)
{
    arParam(dataParam.f_info, dataParam.img, dataParam.scale_info);
}

template <class Archive>
void serialize(Archive& arParam, BatchImageParaWithScaleT& dataParam)
{
    arParam(dataParam.b_info, dataParam.v_img);
}

template <class Archive>
void serialize(Archive& arParam, ImageAll& dataParam)
{
    arParam(dataParam.width_org, dataParam.height_org, dataParam.channal_org, dataParam.image);
}

template <class Archive>
void serialize(Archive& arParam, BatchImageParaScale& dataParam)
{
    arParam(dataParam.b_info, dataParam.v_img);
}

template <class Archive>
void serialize(Archive& arParam, OutputT& dataParam)
{
    arParam(dataParam.size);
    arParam(dataParam.name);
    if (dataParam.size > 0 && dataParam.data.get() == nullptr) {
        dataParam.data.reset(new u_int8_t[dataParam.size]);
    }

    arParam(cereal::binary_data(dataParam.data.get(), dataParam.size * sizeof(u_int8_t)));
}

template <class Archive>
void serialize(Archive& arParam, EngineTransT& dataParam)
{
    arParam(dataParam.status, dataParam.msg, dataParam.b_info, dataParam.size, dataParam.output_data_vec, dataParam.v_img);
}

// The new version of serialize function
void GetEvbImageInfoSearPtr(void *inputPtr, std::string& ctrlStr, uint8_t*& dataPtr, uint32_t& dataLen)
{
    if (inputPtr == nullptr) {
        return;
    }
    EvbImageInfo* imageInfo = (EvbImageInfo*)inputPtr;
    ctrlStr = std::string((char*)inputPtr, sizeof(EvbImageInfo));
    dataPtr = (uint8_t*)imageInfo->pucImageData;
    dataLen = imageInfo->size;
}

// The new version of deserialize function
std::shared_ptr<void> GetEvbImageInfoDearPtr(const char* ctrlPtr, const uint32_t& ctrLen, const uint8_t* dataPtr, const uint32_t& dataLen)
{
    if (ctrlPtr == nullptr) {
        return nullptr;
    }
    EvbImageInfo* imageInfo = (EvbImageInfo*)ctrlPtr;
    std::shared_ptr<BatchImageParaWithScaleT> imageHandle = std::make_shared<BatchImageParaWithScaleT>();
    imageHandle->b_info.frame_ID.push_back(imageInfo->frame_ID);
    imageHandle->b_info.batch_size = imageInfo->batch_size;
    imageHandle->b_info.max_batch_size = imageInfo->max_batch_size;
    imageHandle->b_info.batch_ID = imageInfo->batch_ID;
    imageHandle->b_info.is_first = imageInfo->is_first;
    imageHandle->b_info.is_last = imageInfo->is_last;
    if (IsSentinelImage(imageHandle)) {
        return imageHandle;
    }
    NewImageParaT imgData;
    imgData.img.format = (IMAGEFORMAT)imageInfo->format;
    imgData.img.width = imageInfo->width ;
    imgData.img.height = imageInfo->height;
    imgData.img.size = imageInfo->size; // Get image info size;
    imgData.img.data.reset((uint8_t*)dataPtr, hiai::Graph::ReleaseDataBuffer);
    imageHandle->v_img.push_back(imgData);
    return std::static_pointer_cast<void>(imageHandle);
}

/**
* @brief: check if the imageHandle is a sentinel image
* @[in]: imageHandle, the image to check
* @[return]: bool, true if the image is a sentinel image
*/
bool IsSentinelImage(const std::shared_ptr<BatchImageParaWithScaleT> imageHandle)
{
    if (imageHandle && (int)imageHandle->b_info.batch_ID == -1) {
        return true;
    }
    return false;
}

/**
* @brief: get the result file name from the image name
* @[in]: imgFullPath: the image file path
* @[in]: postFix: the type of the result file
*/
std::string GenTfileName(std::string imgFullPath, std::string postFix)
{
    std::size_t nameCnt = imgFullPath.find_last_of(".");
    std::size_t found = imgFullPath.find_last_of("/\\");
    std::string tfileName = "davinci_" + imgFullPath.substr(found + 1, nameCnt - found - 1) + postFix;
    return tfileName;
}

/**
* @brief: replace '/' to '_'  in name
* @[in]: name, the name to replace
*/
void GetOutputName(std::string& name)
{
    while (true) {
        std::string::size_type pos = name.find("/");
        if (pos != std::string::npos) {
            name.replace(pos, 1, "_");
        } else {
            break;
        }
    }
}

/**
* @brief: get the image information from the info_file generated by dataset engine
* @[in]: info_file: the info file path
* @[in]: postFix: the type of the result file
*/
std::unordered_map<int, ImageInfor> SetImgPredictionCorrelation(std::string info_file, std::string postFix)
{
    std::ifstream fin(info_file.c_str());
    if (!fin) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "fail to open info file.");
    }
    std::unordered_map<int, ImageInfor> result;
    std::string lineContent;

    std::getline(fin, lineContent);
    std::string  datasetName;
    int totalFileNum = 0;
    std::stringstream lineStr1(lineContent);
    lineStr1 >> datasetName >> totalFileNum;

    int format = -2;
    int fileNumByFormat = 0;
    int count = 0;
    while (count < totalFileNum) {
        std::getline(fin, lineContent);
        std::stringstream lineStrFormat0(lineContent);
        lineStrFormat0 >> format >> fileNumByFormat;
        count += fileNumByFormat;
        for (int ind = 0; ind < fileNumByFormat; ind++) {
            std::getline(fin, lineContent);
            int frameId;
            std::string img_fullpath;
            int width, height;
            std::stringstream lineStr(lineContent);
            lineStr >> frameId >> img_fullpath >> width >> height;
            ImageInfor imgInfor;
            imgInfor.tfilename = GenTfileName(img_fullpath, postFix);
            imgInfor.height = height;
            imgInfor.width = width;
            imgInfor.format = format;
            result[frameId] = imgInfor;
        }
    }
    return result;
}

/**
* @brief: get the caffe layer name and index
* @[in]: inName: the name of tensor
* @[in]: index: the index of output tensor
* @[in]: outName: the caffe layer name
*/
void GetLayerName(const std::string inName, std::string& index, std::string& outName)
{
    auto pos = inName.find_last_of("_");
    index = inName.substr(pos + 1);
    outName = inName.substr(0, pos);
    for (int numUnderline = 2; numUnderline > 0; numUnderline--) {
        auto posUnderline = outName.find_first_of("_");
        outName = outName.substr(posUnderline + 1);
    }
}

/**
* @brief: create folder to store the detection results
* the folder name on the host will be "result_files/enginename"
*/
HIAI_StatusT CreateFolder(std::string folderPath, mode_t mode)
{
    int folderExist = access(folderPath.c_str(), W_OK);
    if (-1 == folderExist) {
        if (mkdir(folderPath.c_str(), mode) == -1) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Create %s failed !", folderPath.c_str());
            return HIAI_ERROR;
        }
    }
    return HIAI_OK;
}

/**
* @brief: get the information file path in the dataset folder
* @[in]: value, path configuration string
* @[return]: string, info file path
*/
std::string GetInfoFilePath(const std::string pathConfig)
{
    std::string datainfoPath = pathConfig;
    while (datainfoPath.back() == '/' || datainfoPath.back() == '\\') {
        datainfoPath.pop_back();
    }
    std::size_t tmpInd = datainfoPath.find_last_of("/\\");
    std::string infoFile_ = datainfoPath + "/" + "." + datainfoPath.substr(tmpInd + 1) + "_data.info";
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "info file:%s", infoFile_.c_str());
    return infoFile_;
}

HIAI_REGISTER_DATA_TYPE("EngineTransT", EngineTransT);
HIAI_REGISTER_DATA_TYPE("OutputT", OutputT);
HIAI_REGISTER_DATA_TYPE("ScaleInfoT", ScaleInfoT);
HIAI_REGISTER_DATA_TYPE("NewImageParaT", NewImageParaT);
HIAI_REGISTER_DATA_TYPE("BatchImageParaWithScaleT", BatchImageParaWithScaleT);
HIAI_REGISTER_SERIALIZE_FUNC("EvbImageInfo", EvbImageInfo, GetEvbImageInfoSearPtr, GetEvbImageInfoDearPtr);
