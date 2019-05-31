//
// Created by liguiyuan on 19-4-5.
//
#pragma once
#ifndef MOBILEFACENET_AS_RECOGNIZE_H
#define MOBILEFACENET_AS_RECOGNIZE_H
#include <string>
#include "net.h"
#include <algorithm>

namespace Face {

    typedef struct FaceInfo {
        float score;
        int x[2];
        int y[2];
        float area;
        float regreCoord[4];
        int landmark[10];
    } FaceInfo;

    class Recognize {
    public:
        Recognize(const std::string &model_path);
        ~Recognize();
        void start(ncnn::Mat& ncnn_img, std::vector<float>&feature128);
        void SetThreadNum(int threadNum);
        void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M);
        void warpAffineMatrix(ncnn::Mat src, ncnn::Mat &dst, float *M, int dst_w, int dst_h);
        ncnn::Mat preprocess(ncnn::Mat img, FaceInfo info);
    private:
        void RecogNet(ncnn::Mat& img_);
        ncnn::Net Recognet;
        //ncnn::Mat ncnn_img;
        std::vector<float> feature_out;
        int threadnum = 1;
    };

    double calculSimilar(std::vector<float>& v1, std::vector<float>& v2);
}
#endif //MOBILEFACENET_AS_RECOGNIZE_H
