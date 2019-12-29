/*
 * @License: Copyright (c) Huawei Technologies Co., Ltd. 2012-2019. All rights reserved.
 * @Description: yolov3 yolo layer
 */
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>

using std::string;

// 最大目标数
#define MAX_OBJECT  100
#define MAX_OUTPUT_BOX_NUM 300
// 目标类别数
#define CLASS_NUM 15
#define POST_NMS_SIZE (0.45)
#define POST_THRESH_SIZE (0.25)

#define BATCH_SIZE 1
#define ANCHORS_DIM 3
#define DETECT_LAYER_NUM 3

// 模型输入大小
#define MODEL_WIDTH = 416
#define MODEL_HEIGHT = 416

// yolov3
static float g_biases[18] = {20, 24, 25, 55, 57, 39, 50, 90, 122, 88, 83, 158, 150, 223, 257, 140, 311, 272};
static float g_mask[9] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };

static inline float sigmoid(float x) 
{
    return 1. / (1. + exp(-x)); 
}

typedef struct {
    float x, y, w, h;
} box_rect;

typedef struct Detection {
    box_rect bbox;
    float *prob;
    float objectness;
} Detection;

typedef struct {
    float prob;
    int class_id;
    float x;
    float y;
    float width;
    float height;
} DetectBox;

void activate_array(float *x, const int n)
{
    int i;
    for (i = 0; i < n; ++i) {
    x[i] = sigmoid(x[i]);
    }
}
void free_detections(Detection *dets, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
    }
    free(dets);
}

static int entry_index(int w, int h, int c, int batch, int location, int entry)
{
    int n = location / (w * h);
    int loc = location % (w * h);
    
    return batch * w * h * c + n * w * h * (4 + CLASS_NUM + 1) + entry * w * h + loc;
}
float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    
    return right - left;
}
float box_intersection(box_rect a, box_rect b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) {
        return 0;
    }
    float area = w * h;
    return area;
}

float box_union(box_rect a, box_rect b)
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    
    return u;
}
float box_iou(box_rect a, box_rect b)
{
    return box_intersection(a, b) / box_union(a, b);
}

int nms_comparator_v3(const void *pa, const void *pb)
{
    Detection a = *(Detection *)pa;
    Detection b = *(Detection *)pb;

    float diff = a.objectness - b.objectness;
    if (diff < 0) {
        return 1;
    } else if (diff > 0) {
        return -1;
    }

    return 0;
}
void do_nms_sort(Detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;

    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            Detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        qsort(dets, total, sizeof(Detection), nms_comparator_v3);
        for (i = 0; i < total; ++i) {
            if (dets[i].prob[k] == 0) continue;
            box_rect a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box_rect b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

box_rect get_yolo_box(float *x, float *g_biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box_rect b;
    if (lw == 0 || lh == 0 || w == 0 || h == 0) {
        b.x = 0;
        b.y = 0;
        b.w = 0;
        b.h = 0;
        return b;
    }
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * g_biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * g_biases[2 * n + 1] / h;

    return b;
}

void correct_yolo_boxes(Detection *dets, int n, int w, int h, int netw, int neth)
{
    int new_w = 0;
    int new_h = 0;

    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    } else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    
    for (int i = 0; i < n; ++i) {
        box_rect b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        dets[i].bbox = b;
    }
}
int get_yolo_detections(float* data_begin, int w, int h, int c, int img_w, int img_h, int netw, int neth, float thresh, Detection *dets, int q)
{
    int i, j, n;
    float *predictions = data_begin;
    int count = 0;
    for (i = 0; i < w * h; ++i) {
        int row = i / w;
        int col = i % w;
        for (n = 0; n < ANCHORS_DIM; ++n) {
            int obj_index = entry_index(w, h, c, 0, n * w * h + i, 4);
            float objectness = predictions[obj_index];
            if (objectness <= thresh) {
                continue;
            }

            int box_index = entry_index(w, h, c, 0, n * w * h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, g_biases, g_mask[3 * (DETECT_LAYER_NUM - 1 - q) + n], box_index, col, row, w, h, netw, neth, w * h);
            dets[count].objectness = objectness;

            for (j = 0; j < CLASS_NUM; ++j) {
                int class_index = entry_index(w, h, c, 0, n * w * h + i, 4 + 1 + j);
                float prob = objectness * predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    return count;
}


void get_output_layer_info(int layer, int&w, int&h, int&c, int&offset)
{
        int i = layer; // 第几层
        c = 3 * (CLASS_NUM + 1 + 4);
        if (i == 0) {
            w = 13;
            h = 13;
            offset = 0;
        } else if (i == 1) {
            w = 26; 
            h = 26;
            offset = 13 * 13 * c;
        } else if (i == 2) {
            w = 52;
            h = 52;
            offset = (13 * 13 + 26 * 26)*c;
        }
}

class CYolo {

public:
    
    CYolo()
    {
        dets = NULL;
        detectBox = NULL;
    }
    int Init() 
    {
        dets = (Detection *)calloc(MAX_OBJECT, sizeof(Detection));
        if (dets == NULL) {
            return -1;
        }
        for (int i = 0; i < MAX_OBJECT; ++i) {
            dets[i].prob = (float *)calloc(CLASS_NUM, sizeof(float));
        }

        detectBox = new DetectBox[MAX_OBJECT];
        if (detectBox == NULL) {
            return -1;
        }

        reset_detect();

        return 0;
    }


    ~CYolo()
    {
        free_detections(dets, MAX_OBJECT);
        delete []detectBox;
        
        dets = NULL;
        detectBox = NULL;
    }


    DetectBox *process(float* netout, uint32_t size, int imw, int imh, int *box_num)
    {
        float thresh = POST_THRESH_SIZE;
        float nms = POST_NMS_SIZE;
        int box_nr = 0;
        int input_w = 416;
        int input_h = 416;
    
        int out_layers = DETECT_LAYER_NUM;
        for (int i = 0; i < out_layers; i++) {
            int b = 0;
            int w, h, c = 0;
            int offset = 0;
            get_output_layer_info(i, w, h, c, offset);
    
            float* data_begin = netout + offset;


            for (int n = 0; n < ANCHORS_DIM; ++n) {
                int index = entry_index(w, h, c, b, n * w * h, 0);
                activate_array((float*)data_begin + index, 2 * w * h);

                int index2 = entry_index(w, h, c, b, n * w * h, 4);
                activate_array((float*)data_begin + index2, (1 + CLASS_NUM) * w * h);

            }
                            
            for (int k = 0; k < w * h; ++k) {
                for (int n = 0; n < ANCHORS_DIM; ++n) {
                    int obj_index = entry_index(w, h, c, 0, n * w * h + k, 4);
                        if (data_begin[obj_index] > thresh) {
                            ++box_nr;
                        }
                }
            }
        }
        reset_detect();
    
        Detection* dets_p = dets;

        for (int i = 0; i < out_layers;i++) {
            int b = 0;
            int w, h, c = 0;
            int offset = 0;
            get_output_layer_info(i, w, h, c, offset);
            float* data_begin = netout + offset;
    
            int count = get_yolo_detections(data_begin, w, h, c, imw, imh, input_w, input_h, thresh, dets_p, i);
            dets_p += count;

        }
    
        if (nms) {
            do_nms_sort(dets, box_nr, CLASS_NUM, nms);
        }
            
        int j = 0;
        for (int _v = 0; _v < box_nr; _v++) {
            float max_prob = 0.0f;
            int label = -1;
            for (int k = 0; k < CLASS_NUM; k++) {
                if (dets[_v].prob[k] < 0.25) { // 0.25
                    continue;
                }
                                
                if (dets[_v].prob[k] > max_prob) {
                    max_prob = dets[_v].prob[k];
                    label = k;
                }
            }
            if (-1 == label) {
                continue;
            }
            detectBox[j].x = dets[_v].bbox.x;
            detectBox[j].y = dets[_v].bbox.y;
            detectBox[j].width = dets[_v].bbox.w;
            detectBox[j].height = dets[_v].bbox.h;      
            detectBox[j].class_id = label;
            detectBox[j].prob = max_prob;
            j++;
        }
        *box_num = j;
        
        return GetDetectBox();
    }
    

private:
    Detection *dets = NULL;
    DetectBox *detectBox = NULL;
    DetectBox *GetDetectBox() 
    {
        return detectBox;
    }
    
    int reset_detect()
    {
        for (int i = 0; i < MAX_OBJECT; ++i) {
            int ms = memset_s(dets[i].prob, CLASS_NUM * sizeof(float), 0, CLASS_NUM * sizeof(float));
            if (ms != 0) {
                return -1;
            }
        }

        return 0;
    }
    
};

