//
// Created by Administrator on 2019\5\20 0020.
//

#include "liveness.h"

Liveness::Liveness(const std::string &model_path) {
    std::string param_files = model_path + "/liveness.param";
    std::string bin_files = model_path + "/liveness.bin";
    LivenessNet.load_param(param_files.c_str());
    LivenessNet.load_model(bin_files.c_str());
}

Liveness::~Liveness() {
    LivenessNet.clear();
}

void Liveness::start(ncnn::Mat &ncnn_img, std::vector<float> &cls_scores) {
    //ncnn::resize_bilinear(ncnn_img,img_,32,32);
    ncnn_img.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = LivenessNet.create_extractor();
    ex.set_num_threads(threadnum);
    ex.set_light_mode(true);
    ex.input("conv2d_1_input", ncnn_img);     // input node
    ncnn::Mat out;
    ex.extract("activation_6", out);     // output node
    cls_scores.resize(out.w);
    for (int j = 0; j < cls_scores.size(); j++)
    {
        cls_scores[j] = out[j];
    }
}

void Liveness::SetThreadNum(int threadNum) {
    threadnum = threadNum;
}