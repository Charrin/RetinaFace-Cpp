#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"

int main() {
    extern float pixel_mean[3];
    extern float pixel_std[3];
	std::string param_path =  "./models/retina.param";
	std::string bin_path = "./models/retina.bin";
	ncnn::Net _net;
	_net.load_param(param_path.data());
	_net.load_model(bin_path.data());
    cv::Mat img = cv::imread("./images/test.jpg");
    if(!img.data)
    	printf("load error");


	ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 300, 300);
    cv::resize(img, img, cv::Size(300, 300));

    input.substract_mean_normalize(pixel_mean, pixel_std);
	ncnn::Extractor _extractor = _net.create_extractor();
	_extractor.input("data", input);


    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) { 
    	ncnn::Mat cls;
    	ncnn::Mat reg;
    	ncnn::Mat pts;

        // get blob output
        char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        _extractor.extract(clsname, cls);
        _extractor.extract(regname, reg);
        _extractor.extract(ptsname, pts);

        printf("cls %d %d %d\n", cls.c, cls.h, cls.w);
        printf("reg %d %d %d\n", reg.c, reg.h, reg.w);
        printf("pts %d %d %d\n", pts.c, pts.h, pts.w);

        ac[i].FilterAnchor(cls, reg, pts, proposals);

        printf("stride %d, res size %d\n", _feat_stride_fpn[i], proposals.size());

        for (int r = 0; r < proposals.size(); ++r) {
            proposals[r].print();
        }
    }

    // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);

    printf("final result %d\n", result.size());
    for(int i = 0; i < result.size(); i ++)
    {
        cv::rectangle (img, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(0, 255, 255), 2, 8, 0);
        for (int j = 0; j < result[i].pts.size(); ++j) {
        	cv::circle(img, cv::Point((int)result[i].pts[j].x, (int)result[i].pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
        }
    }
    result[0].print();

    cv::imshow("img", img);
    cv::imwrite("result.jpg", img);
    cv::waitKey(0);
    return 0;
}

