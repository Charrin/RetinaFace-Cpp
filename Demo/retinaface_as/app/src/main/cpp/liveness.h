//
// Created by Administrator on 2019\5\20 0020.
//
#pragma once
#ifndef MOBILEFACENET_AS_LIVENESS_H
#define MOBILEFACENET_AS_LIVENESS_H

#include <string>
#include "net.h"
#include <algorithm>

class Liveness {

public:
    Liveness(const std::string &model_path);
    ~Liveness();
    void start(ncnn::Mat &ncnn_img, std::vector<float>&cls_scores);
    void SetThreadNum(int threadNum);

private:
    ncnn::Net LivenessNet;
    ncnn::Mat img_;
    //std::vector<float> cls_name;
    int threadnum = 1;
    const float norm_vals[3] = { 1/255.0, 1 / 255.0, 1 / 255.0 };
};


#endif //MOBILEFACENET_AS_LIVENESS_H
