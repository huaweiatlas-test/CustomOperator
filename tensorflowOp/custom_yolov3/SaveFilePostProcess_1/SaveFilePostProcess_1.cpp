/**
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-5-30
*/
#include "SaveFilePostProcess_1.h"
#include <hiaiengine/log.h>
#include <vector>
#include <unistd.h>
#include <thread>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>
#include <sstream>
#include <fcntl.h>

HIAI_StatusT SaveFilePostProcess_1::Init(const hiai::AIConfig& config, const  std::vector<hiai::AIModelDescription>& model_desc)
{
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[SaveFilePostProcess_1] start init!");
    if (postprocess_config_ == NULL) {
        postprocess_config_ = std::make_shared<PostprocessConfig>();
    }
    for (int index = 0; index < config.items_size(); ++index) {
        const ::hiai::AIConfigItem& item = config.items(index);
        std::string name = item.name();
        if (name == "path") {
            postprocess_config_->path = item.value();
            break;
        }
    }
    std::string datainfoPath = postprocess_config_->path;
    while (datainfoPath.back() == '/' || datainfoPath.back() == '\\') {
        datainfoPath.pop_back();
    }

    std::size_t tmpInd = datainfoPath.find_last_of("/\\");
    postprocess_config_->info_file = "." + datainfoPath.substr(tmpInd + 1) + "_data.info";
    std::string infoFile = datainfoPath + "/" + postprocess_config_->info_file;
    id_img_correlation.clear();
    char path[PATH_MAX] = {0};
    if (realpath(infoFile.c_str(), path) == NULL) {
        has_data_info_file = false;
        if (datainfoPath.substr(tmpInd + 1) != "MnistDataset") {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[SaveFilePostProcess_1] can not find %s!", postprocess_config_->info_file.c_str());
            return HIAI_ERROR;
        }
    } else {
        has_data_info_file = true;
        id_img_correlation = SetImgPredictionCorrelation(infoFile, "");
    }

    uint32_t graphId = Engine::GetGraphId();
    std::shared_ptr<Graph> graph = Graph::GetInstance(graphId);
    if (nullptr == graph) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Fail to get the graph");
        return HIAI_ERROR;
    }
    std::ostringstream deviceId;
    deviceId << graph->GetDeviceID();
    string deviceDir = RESULT_FOLDER + "/" + deviceId.str();
    store_path = deviceDir + "/" + ENGINE_NAME;
    if (HIAI_OK != CreateFolder(RESULT_FOLDER, PERMISSION)) {
        return HIAI_ERROR;
    }
    if (HIAI_OK != CreateFolder(deviceDir, PERMISSION)) {
        return HIAI_ERROR;
    }
    if (HIAI_OK != CreateFolder(store_path, PERMISSION)) {
        return HIAI_ERROR;
    }
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[SaveFilePostProcess_1] end init!");
    return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("SaveFilePostProcess_1", SaveFilePostProcess_1, INPUT_SIZE)
{
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[SaveFilePostProcess_1] start process!");
    if (nullptr == arg0) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "fail to process invalid message");
        return HIAI_ERROR;
    }
    std::shared_ptr<EngineTransT> tran = std::static_pointer_cast<EngineTransT>(arg0);
    // add sentinel image for showing this data in dataset are all sended, this is last step.
    BatchImageParaWithScaleT image_handle = {tran->b_info, tran->v_img};
    if (IsSentinelImage(std::make_shared<BatchImageParaWithScaleT>(image_handle))) {
        HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[SaveFilePostProcess_1] sentinel image, process over.");
        std::shared_ptr<std::string> result_data(new std::string);
        HIAI_StatusT hiai_ret = HIAI_OK;
        do {
            hiai_ret = SendData(0, "string", std::static_pointer_cast<void>(result_data));
            if (HIAI_OK != hiai_ret) {
                if (HIAI_ENGINE_NULL_POINTER == hiai_ret || HIAI_HDC_SEND_MSG_ERROR == hiai_ret || HIAI_HDC_SEND_ERROR == hiai_ret
                        || HIAI_GRAPH_SRC_PORT_NOT_EXIST == hiai_ret || HIAI_GRAPH_ENGINE_NOT_EXIST == hiai_ret) {
                    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[SaveFilePostProcess_1] SendData error[%d], break.", hiai_ret);
                    break;
                }
                HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[SaveFilePostProcess_1] SendData return value[%d] not OK, sleep 200ms", hiai_ret);
                usleep(SEND_DATA_INTERVAL_MS);

            }
        } while (HIAI_OK != hiai_ret);
        return hiai_ret;
    }
    if (!tran->status) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, tran->msg.c_str());
        return HIAI_ERROR;
    }
    std::vector<OutputT> output_data_vec = tran->output_data_vec;
    std::vector<uint32_t> frame_ID = tran->b_info.frame_ID;
    for (unsigned int ind = 0; ind < tran->b_info.batch_size; ind++) {
        if ((int)frame_ID[ind] == -1) {
            HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[ImageClassificationPostProcess] image number %d failed, add empty struct", ind);
            continue;
        }
        std::string prefix = "";
        if (has_data_info_file) {
            ImageInfor img_infor = id_img_correlation[frame_ID[ind]];
            prefix = store_path  + "/" + img_infor.tfilename;
        } else {
            prefix = store_path  + "/" + std::to_string(frame_ID[ind]);
        }
        for (unsigned int i=0 ; i < output_data_vec.size(); ++i) {
            OutputT out = output_data_vec[i];
            int size = out.size / sizeof(float);
            if (size <= 0) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[SaveFilePostProcess_1]: the OutPutT size less than 0!");
                return HIAI_ERROR;
            }
            float* result = nullptr;
            try {
                result = new float[size];
            } catch (const std::bad_alloc& e) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[SaveFilePostProcess_1] alloc output data error!");
                return HIAI_ERROR;
            }
            int ret  = memcpy_s(result, sizeof(float)*size, out.data.get(), sizeof(float)*size);
            if (ret != 0) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[SaveFilePostProcess_1] memcpy_s output data error!");
                delete[] result;
                result = NULL;
                return HIAI_ERROR;
            }
            std::string name(out.name);
            GetOutputName(name);
            std::string outFileName = prefix + "_" + name + ".txt";
            int fd = open(outFileName.c_str(), O_CREAT| O_WRONLY, FIlE_PERMISSION);
            if (fd == -1) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[SaveFilePostProcess_1] open file %s error!", outFileName.c_str());
                delete[] result;
                result = NULL;
                return HIAI_ERROR;
            }
            int oneResultSize = size / tran->b_info.max_batch_size;
            for (int k = 0; k < oneResultSize; k++) {
                std::string value = std::to_string(result[oneResultSize *ind + k]);
                if (k > 0) {
                    value = "\n" + value;
                }
                ret = write(fd, value.c_str(), value.length());
                if (ret == -1) {
                    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[SaveFilePostProcess_1] write file error!");
                    ret = close(fd);
                    if (ret == -1) {
                        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[SaveFilePostProcess_1] close file error!");
                    }
                    delete[] result;
                    result = NULL;
                    return HIAI_ERROR;
                }
            }
            ret = close(fd);
            if (ret == -1) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[SaveFilePostProcess_1] close file error!");
                delete[] result;
                result = NULL;
                return HIAI_ERROR;
            }
            delete[] result;
            result = NULL;
        }
    }
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[SaveFilePostProcess_1] end process!");
    return HIAI_OK;
}
