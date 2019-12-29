/**
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-5-19
*/
#include <unistd.h>
#include <thread>
#include <fstream>
#include <algorithm>
#include "main.h"
#include "hiaiengine/api.h"
#include <libgen.h>
#include <string.h>


uint32_t g_graphId = 0;
int g_flag = 1;
std::mutex g_mt;
/**
* @ingroup FasterRcnnDataRecvInterface
* @brief RecvData RecvData回调，保存文件
* @param [in]
*/
HIAI_StatusT CustomDataRecvInterface::RecvData(const std::shared_ptr<void>& message)
{
    std::shared_ptr<std::string> data =
        std::static_pointer_cast<std::string>(message);
    g_mt.lock();
    g_flag--;
    g_mt.unlock();
    return HIAI_OK;
}

// if device is disconnected, destroy the graph
HIAI_StatusT DeviceDisconnectCallBack()
{
    g_mt.lock();
    g_flag = 0;
    g_mt.unlock();
    return HIAI_OK;
}

// Init and create graph
HIAI_StatusT HIAI_InitAndStartGraph()
{
    // Step1: Global System Initialization before using HIAI Engine
    HIAI_StatusT status = HIAI_Init(0);

    // Step2: Create and Start the Graph
    std::list<std::shared_ptr<hiai::Graph>> graphList;
    status = hiai::Graph::CreateGraph("./graph.config", graphList);
    if (status != HIAI_OK) {
        HIAI_ENGINE_LOG(status, "Fail to start graph");
        return status;
    }

    // Step3
    std::shared_ptr<hiai::Graph> graph = graphList.front();
    if (nullptr == graph) {
        HIAI_ENGINE_LOG("Fail to get the graph");
        return status;
    }
    g_graphId = graph->GetGraphId();
    int leafArray[1] = {816};  // leaf node id

    for (int i = 0; i < 1; i++) {
        hiai::EnginePortID targetPortConfig;
        targetPortConfig.graph_id = g_graphId;
        targetPortConfig.engine_id = leafArray[i];  
        targetPortConfig.port_id = 0;
        graph->SetDataRecvFunctor(targetPortConfig,
            std::shared_ptr<CustomDataRecvInterface>(new CustomDataRecvInterface("")));
        graph->RegisterEventHandle(hiai::HIAI_DEVICE_DISCONNECT_EVENT,
            DeviceDisconnectCallBack);
    }
    return HIAI_OK;
}
int main(int argc, char* argv[])
{
    char* dirc = strdup(argv[0]);
    if (dirc) {
        char* dname = ::dirname(dirc);
        chdir(dname);
        HIAI_ENGINE_LOG("chdir to %s", dname);
        free(dirc);
    }
    // 1.create graph
    HIAI_StatusT ret = HIAI_InitAndStartGraph();
    if (ret != HIAI_OK) {
        HIAI_ENGINE_LOG("Fail to start graph");;
        return -1;
    }

    // 2.send data
    std::shared_ptr<hiai::Graph> graph = hiai::Graph::GetInstance(g_graphId);
    if (nullptr == graph) {
        HIAI_ENGINE_LOG("Fail to get the graph-%u", g_graphId);
        return -1;
    }
    
    // send data to SourceEngine 0 port 
    hiai::EnginePortID engineId;
    engineId.graph_id = g_graphId;
    engineId.engine_id = 141;  // engine_id in graph_config
    engineId.port_id = 0;
    std::shared_ptr<std::string> srcData(new std::string);
    graph->SendData(engineId, "string", std::static_pointer_cast<void>(srcData));
    for (;;) {
        if (g_flag <= 0) {
            break;
        } else {
            usleep(100000);
        }
    }
    hiai::Graph::DestroyGraph(g_graphId);
    return 0;
}
