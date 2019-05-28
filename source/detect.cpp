#include "anchor_generator.h"
// place your inference header here
// #include "inference.hpp"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"

int main() {

    cv::Mat img = cv::imread("test.jpg"); 
    cv::cvtColor(img, img, CV_BGR2RGB);

    // please replace your own inference code

    /*
    Inference net;
    NetContext* nc;

    char* cfg = "0";
    net.Load("mnet.caffemodel", "mnet.prototxt", cfg, NULL);
    nc = net.getNetContext(false);

    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    net.SetInput(nc, "data", img.data, FROM_8U, 1, img.channels(), img.rows, img.cols, pixel_mean, pixel_std);
    net.Forward(nc); 
    */

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) { 
        Cube cls;
        Cube reg;
        Cube pts;

        // get blob output
        char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);

        net.Extract(nc, clsname, cls);
        net.Extract(nc, regname, reg);
        net.Extract(nc, ptsname, pts);

        printf("cls %d %d %d\n", cls.c(), cls.h(), cls.w());
        printf("reg %d %d %d\n", reg.c(), reg.h(), reg.w());
        printf("pts %d %d %d\n", pts.c(), pts.h(), pts.w());

        ac[i].FilterAnchor(&cls, &reg, &pts, proposals);

        printf("stride %d, res size %d\n", _feat_stride_fpn[i], proposals.size());

        for (int r = 0; r < proposals.size(); ++r) {
            proposals[r].print();
        }
    }

    // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);

    printf("final result %d\n", result.size());
    result[0].print();

    return 0;
}
