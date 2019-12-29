/**
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
* @version 1.0
* @date 2018-5-30
*/
#ifndef SaveFilePostProcess_1_ENGINE_H_
#define SaveFilePostProcess_1_ENGINE_H_
#include "hiaiengine/api.h"
#include "hiaiengine/data_type.h"
#include "hiaiengine/engine.h"
#include "hiaiengine/data_type_reg.h"
#include <map>
#include "BatchImageParaWithScale.h"
#define INPUT_SIZE 1
#define OUTPUT_SIZE 1

using hiai::Engine;
using namespace hiai;

class SaveFilePostProcess_1 : public Engine {
private:
    typedef struct PostprocessConfig_{
        std::string path;
        std::string info_file;
        PostprocessConfig_() : path(""), info_file(""){}
    }PostprocessConfig;

public:
    SaveFilePostProcess_1() : postprocess_config_(NULL),has_data_info_file(true),store_path(""){}
    ~SaveFilePostProcess_1(){}
    HIAI_StatusT Init(const hiai::AIConfig& config, const std::vector<hiai::AIModelDescription>& model_desc);
    /**
    * @ingroup hiaiengine
    * @brief HIAI_DEFINE_PROCESS : Overloading Engine Process processing logic
    * @[in]: Define an input port, an output port */ 
    HIAI_DEFINE_PROCESS(INPUT_SIZE, OUTPUT_SIZE);

private:
    const std::string RESULT_FOLDER = "result_files";
    const std::string ENGINE_NAME = "SaveFilePostProcess_1";
    std::shared_ptr<PostprocessConfig> postprocess_config_;
    bool has_data_info_file;
    std::string store_path;
    std::unordered_map<int, ImageInfor> id_img_correlation;
};

#endif
