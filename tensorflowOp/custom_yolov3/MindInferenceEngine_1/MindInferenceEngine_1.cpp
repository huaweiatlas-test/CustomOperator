/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
    * @version 1.0
    * @date 2018-5-22
 */
#include "MindInferenceEngine_1.h"
#include <hiaiengine/log.h>
#include <hiaiengine/ai_types.h>
#include "hiaiengine/ai_memory.h"
#include <vector>
#include <unistd.h>
#include <thread>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cmath>

const static int IMAGE_INFO_DATA_NUM = 3;
const static int16_t DEFAULT_MEAN_VALUE_CHANNEL_0 = 104;
const static int16_t DEFAULT_MEAN_VALUE_CHANNEL_1 = 117;
const static int16_t DEFAULT_MEAN_VALUE_CHANNEL_2 = 123;
const static int16_t DEFAULT_MEAN_VALUE_CHANNEL_3 = 0;
const static float DEFAULT_RECI_VALUE = 1.0;
const static int32_t CROP_START_LOCATION = 0;
const static int32_t WIDTH_ALIGN = 128;
const static int32_t HEIGHT_ALIGN = 16;
const static int32_t BLANK_SET_NUMBER = 0;
const static std::string DYNAMIC_AIPP = "1";
const static std::string STATIC_AIPP = "0";
const static std::string INPUT_IMAGE_FORMAT_YUV420SP_U8 = "YUV420SP_U8";
const static std::string INPUT_IMAGE_FORMAT_XRGB8888_U8 = "XRGB8888_U8";
const static std::string INPUT_IMAGE_FORMAT_RGB888_U8 = "RGB888_U8";
const static std::string INPUT_IMAGE_FORMAT_YUV400_U8 = "YUV400_U8";
const static std::string MODEL_IMAGE_FORMAT_YUV444SP_U8 = "YUV444SP_U8";
const static std::string MODEL_IMAGE_FORMAT_YVU444SP_U8 = "YVU444SP_U8";
const static std::string MODEL_IMAGE_FORMAT_RGB888_U8 = "RGB888_U8";
const static std::string MODEL_IMAGE_FORMAT_BGR888_U8 = "BGR888_U8";
const static std::string MODEL_IMAGE_FORMAT_GRAY = "GRAY";
const static std::string CSC_SWITCH_ON = "1";
const static uint32_t INPUT_EDGE_INDEX_0 = 0;
const static uint32_t INPUT_INDEX_0 = 0;
const int DMALLOC_TIMEOUT = 1000;
const int TIME_NUM = 1000000;
const int ARRAY_NUM = 3;

/**
* @brief: clear buffer in vector
 */
void MindInferenceEngine_1::ClearOutData()
{
    input_data_vec.clear();
    // release outData pre allocate memmory
    for (auto buffer : m_outData_) {
        if (buffer != nullptr) {
            hiai::HIAIMemory::HIAI_DFree(buffer);
            buffer = nullptr;
        }
    }
    m_outData_.clear();
}

/**
* @brief: init, inherited from hiaiengine lib
 */
HIAI_StatusT MindInferenceEngine_1::Init(const hiai::AIConfig &config,
                                         const std::vector<hiai::AIModelDescription> &model_desc)
{
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1] start init!");

    if (ai_model_manager_ == nullptr) {
        ai_model_manager_ = std::make_shared<hiai::AIModelManager>();
    }

    std::vector<hiai::AIModelDescription> modelDescVec;
    hiai::AIModelDescription modelDesc;
    dynamic_aipp_flag = false;
    for (int index = 0; index < config.items_size(); ++index) {
        const ::hiai::AIConfigItem &item = config.items(index);
        if (item.name() == "model_path") {
            std::string model_path = item.value();
            if (model_path.empty()) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1] model_path not exist!");
                return HIAI_ERROR;
            }
            modelDesc.set_path(model_path);
            std::size_t modelNameStartPos =
                model_path.find_last_of("/\\");
            std::size_t modelNameEndPos = model_path.find_last_of(".");
            if (std::string::npos != modelNameStartPos && std::string::npos != modelNameEndPos
                && modelNameEndPos > modelNameStartPos) {
                modelName_ = model_path.substr(modelNameStartPos + 1, modelNameEndPos - modelNameStartPos - 1);
            }
        } else if (item.name() == "passcode") {
            std::string pCode = item.value();
            modelDesc.set_key(pCode);
        } else if (item.name() == "dynamic_aipp_flag") {
            if (item.value() == STATIC_AIPP) {
                dynamic_aipp_flag = false;
            } else if (item.value() == DYNAMIC_AIPP) {
                dynamic_aipp_flag = true;
            } else {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1] wrong dynamic_aipp_flag value");
                return HIAI_ERROR;
            }
        } else if (item.name() == "input_image_format") {
            if (item.value() == INPUT_IMAGE_FORMAT_YUV420SP_U8) {
                input_image_format = hiai::YUV420SP_U8;
            } else if (item.value() == INPUT_IMAGE_FORMAT_XRGB8888_U8) {
                input_image_format = hiai::XRGB8888_U8;
            } else if (item.value() == INPUT_IMAGE_FORMAT_RGB888_U8) {
                input_image_format = hiai::RGB888_U8;
            } else if (item.value() == INPUT_IMAGE_FORMAT_YUV400_U8) {
                input_image_format = hiai::YUV400_U8;
            } else {
                input_image_format = hiai::YUV420SP_U8;
            }
        } else if (item.name() == "csc_switch") {
            if (item.value() == CSC_SWITCH_ON) {
                csc_switch = true;
            } else {
                csc_switch = false;
            }
        } else if (item.name() == "model_image_format") {
            if (item.value() == MODEL_IMAGE_FORMAT_YUV444SP_U8) {
                model_image_format = hiai::MODEL_YUV444SP_U8;
            } else if (item.value() == MODEL_IMAGE_FORMAT_YVU444SP_U8) {
                model_image_format = hiai::MODEL_YVU444SP_U8;
            } else if (item.value() == MODEL_IMAGE_FORMAT_RGB888_U8) {
                model_image_format = hiai::MODEL_RGB888_U8;
            } else if (item.value() == MODEL_IMAGE_FORMAT_BGR888_U8) {
                model_image_format = hiai::MODEL_BGR888_U8;
            } else if (item.value() == MODEL_IMAGE_FORMAT_GRAY) {
                model_image_format = hiai::MODEL_GRAY;
            } else {
                model_image_format = hiai::MODEL_BGR888_U8;
            }
        } else if (item.name() == "input_image_width") {
            if (item.value() == "") {
                input_image_width = BLANK_SET_NUMBER;
                crop_width = BLANK_SET_NUMBER;
            } else {
                std::stringstream ss(item.value());
                ss >> input_image_width;
                if (input_image_width <= 0) {
                    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1] input_image_width <= 0!");
                    return HIAI_ERROR;
                }
                crop_width = input_image_width;
                input_image_width = ceil(input_image_width * 1.0 / WIDTH_ALIGN) * WIDTH_ALIGN;
            }
            
        } else if (item.name() == "input_image_height") {
            if (item.value() == "") {
                input_image_height = BLANK_SET_NUMBER;
                crop_height = BLANK_SET_NUMBER;
            } else {
                std::stringstream ss(item.value());
                ss >> input_image_height;
                if (input_image_height <= 0) {
                    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1] input_image_height <= 0!");
                    return HIAI_ERROR;
                }
                crop_height = input_image_height;
                input_image_height = ceil(input_image_height * 1.0 / HEIGHT_ALIGN) * HEIGHT_ALIGN;
            }
        }
    }
    modelDesc.set_name(modelName_);
    modelDescVec.push_back(modelDesc);
    hiai::AIStatus ret = ai_model_manager_->Init(config, modelDescVec);
    if (ret != hiai::SUCCESS) {
        return HIAI_ERROR;
    }

    ret = ai_model_manager_->GetModelIOTensorDim(modelName_, inputTensorVec, outputTensorVec);
    if (ret != hiai::SUCCESS) {
        HIAI_ENGINE_LOG(this, HIAI_IDE_ERROR, "hiai ai model manager init fail");
        return HIAI_ERROR;
    }
    m_detectBox = new DetectBox[MAX_OUTPUT_BOX_NUM];
    batch_size = inputTensorVec[INPUT_INDEX_0].n;
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1] end init!");
    return HIAI_OK;
}

MindInferenceEngine_1::~MindInferenceEngine_1()
{
    if (m_detectBox != NULL) {
        delete [] m_detectBox;
        m_detectBox = NULL;
    }

    if (m_yolo != NULL) {
        delete m_yolo;
        m_yolo = NULL;
    }
}

/**
* @brief: handle the exceptions when the dataset batch failed
* @in: errorMsg: the error message
*/
void MindInferenceEngine_1::HandleExceptions(std::string errorMsg)
{
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, errorMsg.c_str());
    tran_data->status = false;
    tran_data->msg = errorMsg;
    // send null to next node to avoid blocking when to encounter abnomal situation.
    auto ret = SendData(0, "EngineTransT", std::static_pointer_cast<void>(tran_data));
    if (ret != HIAI_OK) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1] send data failed!");
    }
};

// send sentinel image to inform the graph to destroy
HIAI_StatusT MindInferenceEngine_1::SendSentinelImage() 
{
    tran_data->status = true;
    tran_data->msg = "sentinel Image";
    tran_data->b_info = image_handle->b_info;
    HIAI_StatusT hiaiRet = HIAI_OK;
    do {
        HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1]
                                        sentinel image, process
                                        success!");
        hiaiRet = SendData(0, "EngineTransT", std::static_pointer_cast<void>(tran_data));
        if (hiaiRet != HIAI_OK) {
            if (hiaiRet == HIAI_ENGINE_NULL_POINTER || hiaiRet == HIAI_HDC_SEND_MSG_ERROR || hiaiRet == HIAI_HDC_SEND_ERROR 
                || hiaiRet == HIAI_GRAPH_SRC_PORT_NOT_EXIST || hiaiRet == HIAI_GRAPH_ENGINE_NOT_EXIST) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1]
                                        SendData error[%d],
                                        break.", hiaiRet);
                break;
            }
            HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1]
                                        SendData return value[%d] not OK, sleep
                                        200ms", hiaiRet);
            usleep(SEND_DATA_INTERVAL_MS);
        }
    } while (hiaiRet != HIAI_OK);
    return hiaiRet;
}

/**
* @brief: prepare the data buffer for image information
* @in: input_buffer: buffer pointer
* @in: imageNumber: total number of received images
* @in: batchBegin: the index of the first image of each batch
* @in: image_size: size of each image
* @return: HIAI_StatusT
*/
HIAI_StatusT MindInferenceEngine_1::PrepareInputBuffer(uint8_t *input_buffer, const int imageNumber, const int batchBegin, const int image_size)
{
    // 1.prepare input buffer for each batch
    // the loop for each image
    if (input_buffer ==  nullptr) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1]
                                        ERROR, input_buffer is
                                        nullptr");
        return HIAI_ERROR;
    }
    for (int j = 0; j < batch_size; j++) {
        if (batchBegin + j < imageNumber) {
            if (memcpy_s(input_buffer + j * image_size, image_size, image_handle->v_img[batchBegin + j].img.data.get(), image_size)) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1]
                                        ERROR, copy image buffer
                                        failed");
                return HIAI_ERROR;
            }
        } else {
            if (memset_s(input_buffer + j * image_size, image_size, static_cast<char>(0), image_size)) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1]
                                        ERROR, batch padding for image data
                                        failed");
                return HIAI_ERROR;
            }
        }
    }
    return HIAI_OK;
}

/**
* @brief: prepare the data buffer for image information
* @in: input_buffer2: buffer pointer
* @in: imageNumber: total number of received images
* @in: batchBegin: the index of the first image of each batch
* @in: multi_input_2: the second input received from the previous engine
* @return: HIAI_StatusT
*/
HIAI_StatusT MindInferenceEngine_1::PrepareInforInput(uint8_t *input_buffer2, const int imageNumber, const int batchBegin, std::shared_ptr<hiai::BatchRawDataBuffer> multi_input_2)
{
    int eachSize;
    // the loop for each info
    for (int j = 0; j < batch_size; j++) {
        if (batchBegin + j < imageNumber) {
            hiai::RawDataBuffer _input_arg_2 = multi_input_2->v_info[batchBegin + j];
            eachSize = _input_arg_2.len_of_byte;
            HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1]
                                        info each input size:
                                        %d", eachSize);
            if (memcpy_s(input_buffer2 + j * eachSize, eachSize, _input_arg_2.data.get(), eachSize)) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1]
                                        ERROR, copy info buffer
                                        failed");
                return HIAI_ERROR;
            }
        } else {
            float info_tmp[ARRAY_NUM] = { 0.0, 0.0, 0.0 };
            eachSize = sizeof(info_tmp);
            HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1]
                                        info padding size:
                                        %d", eachSize);
            if (memcpy_s(input_buffer2 + j*eachSize, eachSize, info_tmp, eachSize)) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1]
                                        ERROR, padding info buffer
                                        failed");
                return HIAI_ERROR;
            }
        }
    }
    return HIAI_OK;
}


// call ai model manager to do the prediction
HIAI_StatusT MindInferenceEngine_1::Predict()
{
    // pre malloc OutData
    HIAI_StatusT hiaiRet = HIAI_OK;
    for (uint32_t index = 0; index < outputTensorVec.size(); index++) {
        hiai::AITensorDescription outputTensorDesc = hiai::AINeuralNetworkBuffer::GetDescription();
        uint8_t* buf = nullptr;
        hiaiRet = hiai::HIAIMemory::HIAI_DMalloc(outputTensorVec[index].size, (void *&)buf, DMALLOC_TIMEOUT);
        if (hiaiRet != HIAI_OK || buf == nullptr) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1] HIAI_DMalloc failed.");
            ClearOutData();
            return HIAI_ERROR;
        }
        m_outData_.push_back(buf);
        std::shared_ptr<hiai::IAITensor> outputTensor = hiai::AITensorFactory::GetInstance()->CreateTensor(outputTensorDesc, buf, outputTensorVec[index].size);
        shared_ptr<hiai::AINeuralNetworkBuffer> nn_tensor = static_pointer_cast<hiai::AINeuralNetworkBuffer>(outputTensor);
        nn_tensor->SetName(outputTensorVec[index].name);
        output_data_vec.push_back(outputTensor);
    }

    // put buffer to FrameWork directly, InputSize has only one
    hiai::AITensorDescription inputTensorDesc = hiai::AINeuralNetworkBuffer::GetDescription();
    for (unsigned int i = 0; i < predict_input_data_.size(); i++) {
        std::map<uint8_t *, int> tmp = predict_input_data_[i];
        for (std::map<uint8_t *, int>::iterator it = tmp.begin();it != tmp.end(); ++it) {
            shared_ptr<hiai::IAITensor> inputTensor =
                hiai::AITensorFactory::GetInstance()->CreateTensor(inputTensorDesc, reinterpret_cast<void *>(it->first), it->second);
            input_data_vec.push_back(inputTensor); // AIModelManager push input data
        }
    }

    HIAI_StatusT ret = HIAI_OK;
    // dynamic aipp settings
    if (dynamic_aipp_flag) {
        stringstream ss;
        ss << batch_size;
        string batchSizeString = ss.str();
        hiai::AITensorDescription dynamicParam =  hiai::AippDynamicParaTensor::GetDescription(batchSizeString);
        shared_ptr<hiai::IAITensor> tmp_tensor = hiai::AITensorFactory::GetInstance()->CreateTensor(dynamicParam);
        shared_ptr<hiai::AippDynamicParaTensor> dynamicParamTensor = std::static_pointer_cast<hiai::AippDynamicParaTensor>(tmp_tensor);

        // if there are multiple input, we can set multiple input or input edge, default 0
        dynamicParamTensor->SetDynamicInputEdgeIndex(INPUT_EDGE_INDEX_0);
        dynamicParamTensor->SetDynamicInputIndex(INPUT_INDEX_0);

        // set input format
        dynamicParamTensor->SetInputFormat(input_image_format);

        // set csc params if csc switch is true
        if (csc_switch) {
            dynamicParamTensor->SetCscParams(input_image_format, model_image_format, hiai::JPEG);
        }

        // If use image preprocess, set src image size
        if (input_image_width > BLANK_SET_NUMBER && input_image_height > BLANK_SET_NUMBER) {
            dynamicParamTensor->SetSrcImageSize(input_image_width, input_image_height);
        }

        // Every image of a batch can set these properties below independently
        for (int batch_index = 0; batch_index < batch_size; batch_index++) {
            // set default crop, we can set it customize
            if (crop_width > BLANK_SET_NUMBER && crop_height > BLANK_SET_NUMBER) {
                dynamicParamTensor->SetCropParams(true,
                                                  CROP_START_LOCATION,
                                                  CROP_START_LOCATION,
                                                  crop_width,
                                                  crop_height,
                                                  batch_index);
            }

            // set default mean value, we can set it customize
            dynamicParamTensor->SetDtcPixelMin(DEFAULT_MEAN_VALUE_CHANNEL_0,
                                               DEFAULT_MEAN_VALUE_CHANNEL_1,
                                               DEFAULT_MEAN_VALUE_CHANNEL_2,
                                               DEFAULT_MEAN_VALUE_CHANNEL_3,
                                               batch_index);

            // set default dtcPixelVarReci value, we can set it customize
            dynamicParamTensor->SetPixelVarReci(DEFAULT_RECI_VALUE,
                                                DEFAULT_RECI_VALUE,
                                                DEFAULT_RECI_VALUE,
                                                DEFAULT_RECI_VALUE,
                                                batch_index);
        }

        ret = ai_model_manager_->SetInputDynamicAIPP(input_data_vec, dynamicParamTensor);
        if (ret != hiai::SUCCESS) {
            HIAI_ENGINE_LOG(this, HIAI_IDE_ERROR, "hiai set input dynamic aipp fail");
            return HIAI_ERROR;
        }
    }

    hiai::AIContext aiContext;
    ret = ai_model_manager_->Process(aiContext, input_data_vec, output_data_vec, 0);
    if (ret != hiai::SUCCESS) {
        ClearOutData();
        return HIAI_ERROR;
    }
    
    static double total = 0.0;
    static unsigned int cnt = 0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    uint32_t allSize = 0;
    uint32_t singleSize = 0;
    int i = 0;
    int j = 0;
    int batch = batch_size;
    if (!batch) {
        batch = 1;
    }

    for (j = 0; j < output_data_vec.size(); j++) {
        std::shared_ptr<hiai::AISimpleTensor> result_tensor = std::static_pointer_cast<hiai::AISimpleTensor>(output_data_vec[j]);
        allSize += result_tensor->GetSize();
        singleSize += result_tensor->GetSize() / batch;
    }

    uint32_t idx = 0;
    uint32_t offset = 0;
    int height = 416;
    int width = 416;
    int boxNum = 0;
    int allBoxNum = 0;
    DetectBox *detectBox = NULL;

    if (m_yolo == NULL) {
        m_yolo = new CYolo;
        int retYolo = m_yolo->Init();
        if (retYolo == -1) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[FaceDetectionEngine] YOLO init ERROR!");
            return HIAI_ERROR;
        }
    }
    
    unsigned char *singleResult = new unsigned char[singleSize];
    // #pragma omp parallel for
    for (i = 0; i < batch; i++) {
        idx = 0;
        // #pragma omp parallel for
        for (j = 0; j < output_data_vec.size(); j++) {
            std::shared_ptr<hiai::AISimpleTensor> result_tensor = std::static_pointer_cast<hiai::AISimpleTensor>(output_data_vec[j]);
            offset = i * result_tensor->GetSize() / batch;
            ret = memcpy_s(singleResult + idx, result_tensor->GetSize() / batch, result_tensor->GetBuffer() + offset, result_tensor->GetSize() / batch);
            if (ret) {
                HIAI_ENGINE_LOG("memcpy_s
                                        failed! ");
                delete [] singleResult;
                singleResult = NULL;
                return HIAI_ERROR;
            }
            idx += result_tensor->GetSize() / batch;
        }
        
        detectBox = m_yolo->process(reinterpret_cast<float*>(singleResult), singleSize, height, width, &boxNum);

        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "boxNum
                                        : %d", boxNum);
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "detectBox2:
                                        %x", detectBox);
        
        for (int i = 0; i < boxNum; ++i) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "box_%d
                                        : x:%f, y:%f, width:%f, height:%f, label:%d,
                                        prob:%f", i, detectBox[i].x, detectBox[i].y, detectBox[i].width, detectBox[i].height, detectBox[i].class_id, detectBox[i].prob);
        }

        if (detectBox == NULL) {
            HIAI_ENGINE_LOG("detectBox is NULL ");
            delete [] singleResult;
            singleResult = NULL;
            return HIAI_ERROR;
        }

        if (boxNum != 0) {
            ret = memcpy_s(m_detectBox + allBoxNum, boxNum * sizeof(DetectBox), reinterpret_cast<void*>(detectBox), boxNum * sizeof(DetectBox));
            if (ret) {
                HIAI_ENGINE_LOG("memcpy_s failed ");
                delete m_detectBox;
                m_detectBox = NULL;
                delete [] singleResult;
                singleResult = NULL;
                return HIAI_ERROR;
            }
        }

        allBoxNum += boxNum;
    }

    delete [] singleResult;
	
    gettimeofday(&end, NULL);
    double diff = (end.tv_sec - start.tv_sec) * TIME_NUM + (end.tv_usec - start.tv_usec);
    double avg = 0.0;
    cnt++;
    if (cnt == 1) {
        total = 0.0;
    } else {
        total += diff;
        avg = total / (cnt - 1);
    }
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[Inference Cost] diff time: %f us, avg time: %f us, total time: %f us, count?%d", diff, avg, total, cnt);
    return HIAI_OK;
}

/**
* @brief: set the tran_data with the result of this batch
* @in: index of the begin of this batch
* @return: HIAI_StatusT
*/
HIAI_StatusT MindInferenceEngine_1::SetOutputStruct(const int batchBegin)
{
    for (unsigned int i = 0; i < output_data_vec.size(); ++i) {
        HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1]
                                        build: %d", i);
        std::shared_ptr<hiai::AINeuralNetworkBuffer> result_tensor = std::static_pointer_cast<hiai::AINeuralNetworkBuffer>(output_data_vec[i]);
        auto tensor_size = result_tensor->GetSize();
        if (memcpy_s(tran_data->output_data_vec[i].data.get() + batchBegin / batch_size * tensor_size, tensor_size, result_tensor->GetBuffer(), tensor_size)) {
            return HIAI_ERROR;
        }
        HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1]
                                        build: %d, number:
                                        %d", tensor_size, batchBegin / batch_size * tensor_size);
    }
    return HIAI_OK;
}

/**
* @brief: send the predicted result for one batch
*/
void MindInferenceEngine_1::SendResult()
{
    HIAI_StatusT hiaiRet = HIAI_OK;
    do {
        hiaiRet = SendData(0, "EngineTransT", std::static_pointer_cast<void>(tran_data));
        if (hiaiRet == HIAI_QUEUE_FULL) {
            HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1]
                                        queue full, sleep
                                        200ms");
            usleep(200000);
        }
    } while (hiaiRet == HIAI_QUEUE_FULL);
    if (hiaiRet != HIAI_OK) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1]
                                        SendData failed! error code:
                                        %d", hiaiRet);
    }
}

/**
* @brief: set the frame ID as -1 to indicate this model batch failed
* @in: index of the begin of this batch
*/
void MindInferenceEngine_1::HandleModelBatchFailure(const int batchBegin, const int imageNumber)
{
    for (int i = 0; i < batch_size; i++) {
        if (batchBegin + i < imageNumber) {
            tran_data->b_info.frame_ID[i + batchBegin] = -1;
        }
    }
}

/**
* @ingroup hiaiengine
* @brief HIAI_DEFINE_PROCESS : Realize the port input/output processing
* @[in]: Define an input port, an output port,
*        And the Engine is registered, its called "HIAIMultiEngineExample"
*/
HIAI_IMPL_ENGINE_PROCESS("MindInferenceEngine_1", MindInferenceEngine_1, INPUT_SIZE)
{
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1] start process!");
    HIAI_StatusT hiaiRet = HIAI_OK;
    std::lock_guard<std::mutex> lk(memoryRecursiveMutex_);
    if (tran_data == nullptr) {
        tran_data = std::make_shared<EngineTransT>();
    }
    // 1.PreProcess:Framework input data
    if (arg0 != nullptr)
    {
        std::shared_ptr<BatchImageParaWithScaleT> dataInput = std::static_pointer_cast<BatchImageParaWithScaleT>(arg0);

        if (!IsSentinelImage(dataInput)) {
            if (dataInputIn_ != nullptr) {
                if (dataInputIn_->b_info.batch_ID == dataInput->b_info.batch_ID && !dataInput->v_img.empty() && !dataInput->b_info.frame_ID.empty()) {
                    dataInputIn_->v_img.push_back(dataInput->v_img[0]);
                    dataInputIn_->b_info.frame_ID.push_back(dataInput->b_info.frame_ID[0]);
                }
            } else {
                dataInputIn_ = std::make_shared<BatchImageParaWithScaleT>();
                if (dataInputIn_ == nullptr){
                    HIAI_ENGINE_LOG(HIAI_IDE_WARNING, "[MindInferenceEngine_1] malloc error");
                    return HIAI_ERROR;
                }
                for (int i = 0; i < dataInput->b_info.frame_ID.size(); ++i){
                    dataInputIn_->b_info.frame_ID.push_back(dataInput->b_info.frame_ID[i]);
                }
                dataInputIn_->b_info.batch_size = dataInput->b_info.batch_size;
                dataInputIn_->b_info.max_batch_size = dataInput->b_info.max_batch_size;
                dataInputIn_->b_info.batch_ID = dataInput->b_info.batch_ID;
                dataInputIn_->b_info.is_first = dataInput->b_info.is_first;
                dataInputIn_->b_info.is_last = dataInput->b_info.is_last;
                for (int i = 0; i < dataInput->v_img.size(); ++i) {
                    dataInputIn_->v_img.push_back(dataInput->v_img[i]);
                }
            }
            if (dataInputIn_->v_img.size() != dataInputIn_->b_info.batch_size) {
                HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1] Wait for other %d batch imageinfo!", 
                    (dataInputIn_->b_info.batch_size - dataInputIn_->v_img.size()));
                return HIAI_OK;
            }
            input_que_.PushData(0, dataInputIn_);
            dataInputIn_ = nullptr;
        } else {
            input_que_.PushData(0, arg0);
        }
    }

    image_handle = nullptr;

#if INPUT_SIZE < 3
    if (!input_que_.PopAllData(image_handle)) \
    {
        HandleExceptions("[MindInferenceEngine_1] fail to PopAllData");
        return HIAI_ERROR;
    }
#endif

#if (INPUT_SIZE == 3)
    DEFINE_MULTI_INPUT_ARGS_POP(3);
#endif

#if (INPUT_SIZE == 4)
    DEFINE_MULTI_INPUT_ARGS_POP(4);
#endif

#if (INPUT_SIZE == 5)
    DEFINE_MULTI_INPUT_ARGS_POP(5);
#endif

#if (INPUT_SIZE == 6)
    DEFINE_MULTI_INPUT_ARGS_POP(6);
#endif

#if (INPUT_SIZE == 7)
    DEFINE_MULTI_INPUT_ARGS_POP(7);
#endif

#if (INPUT_SIZE == 8)
    DEFINE_MULTI_INPUT_ARGS_POP(8);
#endif

#if (INPUT_SIZE == 9)
    DEFINE_MULTI_INPUT_ARGS_POP(9);
#endif

#if (INPUT_SIZE == 10)
    DEFINE_MULTI_INPUT_ARGS_POP(10);
#endif

    if (nullptr == image_handle) {
        HandleExceptions("[MindInferenceEngine_1] Image_handle is nullptr");
        return HIAI_ERROR;
    }
    // add sentinel image for showing this data in dataset are all sended, this is last step.
    if (IsSentinelImage(image_handle)) {
        return SendSentinelImage();
    }

    int imageNumber = image_handle->v_img.size();
#if (INPUT_SIZE == 3)
    if (nullptr == _multi_input_2) {
        HandleExceptions("[MindInferenceEngine_1]
                                            fail to process invalid
                                            message");
        return HIAI_ERROR;
    }
    int info_number = _multi_input_2->v_info.size();
    if (info_number != imageNumber) {
        HandleExceptions("[MindInferenceEngine_1]
                                            ERROR the number of image data and
                                            information data
                                            doesn't match!");
    }
    int _input_buffer2_size = sizeof(float) * IMAGE_INFO_DATA_NUM * batch_size;
    uint8_t * _input_buffer2 = nullptr;
    hiaiRet = hiai::HIAIMemory::HIAI_DMalloc(_input_buffer2_size, (void *&)_input_buffer2, 1000);
    if (hiaiRet != HIAI_OK || _input_buffer2 == nullptr) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1] HIAI_DMalloc _input_buffer2 failed.");
        return HIAI_ERROR;
    }
#endif

    int image_size = image_handle->v_img[0].img.size * sizeof(uint8_t);
    int _input_buffer1_size = image_size * batch_size;
    if (_input_buffer1_size <= 0) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1] _input_buffer1_size <= 0");
        return HIAI_ERROR;
    }
    uint8_t *_input_buffer1 = nullptr;
    hiaiRet = hiai::HIAIMemory::HIAI_DMalloc(_input_buffer1_size, (void *&)_input_buffer1, 1000);
    if (hiaiRet != HIAI_OK || _input_buffer1 == nullptr) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1] HIAI_DMalloc _input_buffer1 failed.");
#if (INPUT_SIZE == 3)
        hiai::HIAIMemory::HIAI_DFree(_input_buffer2);
        _input_buffer2 = nullptr;
#endif
        return HIAI_ERROR;
    }

    int cnt_batch = image_handle->b_info.batch_size / batch_size;
    if (image_handle->b_info.batch_size % batch_size != 0) {
        cnt_batch ++;
    }

    tran_data->b_info = image_handle->b_info;
    tran_data->v_img = image_handle->v_img;
    tran_data->status = true;
    tran_data->b_info.max_batch_size = cnt_batch* batch_size;

    // the loop for each batch
    for (int i = 0; i < imageNumber; i += batch_size) {
        predict_input_data_.clear();
        // 1.prepare input buffer for each batch
        if (HIAI_OK != PrepareInputBuffer(_input_buffer1, imageNumber, i, image_size)) {
            HandleModelBatchFailure(i, imageNumber);
            continue;
        }
        std::map<uint8_t *, int> input1;
        input1.insert(std::make_pair(_input_buffer1, _input_buffer1_size));
        predict_input_data_.push_back(input1);
#if (INPUT_SIZE == 2)
        DEFINE_MULTI_INPUT_ARGS(2);
#endif

#if (INPUT_SIZE == 3)
        // int eachSize;
        if (HIAI_OK != PrepareInforInput(_input_buffer2, imageNumber, i, _multi_input_2)) {
            HandleModelBatchFailure(i, imageNumber);
            continue;
        }
        std::map<uint8_t *, int> input2;
        input2.insert(std::make_pair(_input_buffer2, _input_buffer2_size));
        predict_input_data_.push_back(input2);
        DEFINE_MULTI_INPUT_ARGS(3);
#endif

        // 2.Call Process, Predict
        input_data_vec.clear();
        if (HIAI_OK != Predict()) {
            output_data_vec.clear();
            HandleModelBatchFailure(i, imageNumber);
            continue;
        }
        // init the output buffer for one dataset batch(might be multiple model batches)
        if (tran_data->output_data_vec.empty()) {
            tran_data->size = output_data_vec.size();
            HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1] alloc memory for dataset batch, number of outputs of the network: %d", output_data_vec.size());
            for (unsigned int i = 0; i < output_data_vec.size(); i++) {
                OutputT out;
                std::shared_ptr<hiai::AINeuralNetworkBuffer> result_tensor = std::static_pointer_cast<hiai::AINeuralNetworkBuffer>(output_data_vec[i]);
                int buffer_size = result_tensor->GetSize();
                out.name = result_tensor->GetName();
                out.size = buffer_size * cnt_batch;
                if (out.size <= 0) {
                    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[MindInferenceEngine_1] out.size <= 0");
                    hiai::HIAIMemory::HIAI_DFree(_input_buffer1);
                    _input_buffer1 = nullptr;
#if (INPUT_SIZE == 3)
                    hiai::HIAIMemory::HIAI_DFree(_input_buffer2);
                    _input_buffer2 = nullptr;
#endif
                    ClearOutData();
                    return HIAI_ERROR;
                }
                u_int8_t *ptr = nullptr;
                try {
                    ptr = new u_int8_t[out.size];
                }
                catch (const std::bad_alloc& e) {
                    hiai::HIAIMemory::HIAI_DFree(_input_buffer1);
                    _input_buffer1 = nullptr;
#if (INPUT_SIZE == 3)
                    hiai::HIAIMemory::HIAI_DFree(_input_buffer2);
                    _input_buffer2 = nullptr;
#endif
                    ClearOutData();
                    return HIAI_ERROR;
                }
                out.data.reset(ptr);
                tran_data->output_data_vec.push_back(out);
                HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1] cnt_model_batch: %d, image number: %d!", cnt_batch, imageNumber);
            }
        }

        // 3.set the tran_data with the result of this batch
        if (HIAI_OK != SetOutputStruct(i)) {
            ClearOutData();
            output_data_vec.clear();
            HandleModelBatchFailure(i, imageNumber);
            continue;
        }
        output_data_vec.clear();
    }
    SendResult();
    // 6. release sources
    hiai::HIAIMemory::HIAI_DFree(_input_buffer1);
    _input_buffer1 = nullptr;
#if (INPUT_SIZE == 3)
    hiai::HIAIMemory::HIAI_DFree(_input_buffer2);
    _input_buffer2 = nullptr;
#endif
    ClearOutData();
    tran_data = nullptr;
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[MindInferenceEngine_1] end process!");
    return HIAI_OK;
}
