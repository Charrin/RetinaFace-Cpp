//
// Created by Administrator on 2019\5\31 0031.
//

#ifndef RETINAFACE_AS_RETINAFACE_H
#define RETINAFACE_AS_RETINAFACE_H


#include "tools.h"

extern float pixel_mean[3];
extern float pixel_std[3];


class RetinaFace {

public:
    float scale = 0.25;
    RetinaFace(const std::string &model_path);
    RetinaFace(const std::vector<std::string> param_files, const std::vector<std::string> bin_files);
    ~RetinaFace();
    void detect(ncnn::Mat& img_, std::vector<Anchor>& finalBbox);
private:
    ncnn::Mat cls;
    ncnn::Mat reg;
    ncnn::Mat pts;

    ncnn::Mat img;

    ncnn::Net _net;
    //float scale = 0.33;

    std::vector<AnchorGenerator> ac;
    std::vector<Anchor> proposals;
    char clsname[100];
    char regname[100];
    char ptsname[100];
};


#endif //RETINAFACE_AS_RETINAFACE_H
