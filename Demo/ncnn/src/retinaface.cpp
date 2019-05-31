//
// Created by Administrator on 2019\5\31 0031.
//

#include "retinaface.h"

RetinaFace::RetinaFace(const std::string &model_path) {

    std::vector<std::string> param_files = {
            model_path+"/retina.param",
    };

    std::vector<std::string> bin_files = {
            model_path+"/retina.bin",
    };

    _net.load_param(param_files[0].data());
    _net.load_model(bin_files[0].data());
    ac.resize(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }
}

RetinaFace::RetinaFace(const std::vector<std::string> param_files, const std::vector<std::string> bin_files){
    _net.load_param(param_files[0].data());
    _net.load_model(bin_files[0].data());
}

RetinaFace::~RetinaFace(){
    _net.clear();
}

void RetinaFace::detect(ncnn::Mat& img_, std::vector<Anchor>& result)
{
    ncnn::resize_bilinear(img_,img,img_.w*scale,img_.h*scale);
    img.substract_mean_normalize(pixel_mean, pixel_std);
    ncnn::Extractor _extractor = _net.create_extractor();
    _extractor.set_light_mode(true);
    _extractor.set_num_threads(1);
    _extractor.input("data", img);

    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {


        // get blob output
        sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        _extractor.extract(clsname, cls);
        _extractor.extract(regname, reg);
        _extractor.extract(ptsname, pts);



        //printf("cls %d %d %d\n", cls.c, cls.h, cls.w);
        //printf("reg %d %d %d\n", reg.c, reg.h, reg.w);
        //printf("pts %d %d %d\n", pts.c, pts.h, pts.w);

        ac[i].FilterAnchor(cls, reg, pts, proposals);

        //printf("stride %d, res size %d\n", _feat_stride_fpn[i], proposals.size());
        /*
        for (int r = 0; r < proposals.size(); ++r) {
            proposals[r].print();
        }
        */
    }

    // nms
    nms_cpu(proposals, nms_threshold, result);

	for (int i = 0; i < result.size(); i++)
	{
		result[i].anchor.x = result[i].anchor.x / this->scale;
		result[i].anchor.y = result[i].anchor.y / this->scale;
		result[i].anchor.width = result[i].anchor.width / this->scale;
		result[i].anchor.height = result[i].anchor.height / this->scale;

		result[i].finalbox.x = result[i].finalbox.x / this->scale;
		result[i].finalbox.y = result[i].finalbox.y / this->scale;
		result[i].finalbox.width = result[i].finalbox.width / this->scale;
		result[i].finalbox.height = result[i].finalbox.height / this->scale;

		result[i].center.x = result[i].center.x / this->scale;
		result[i].center.y = result[i].center.y / this->scale;

		for (int k = 0; k < 4; k++)
		{
			result[i].reg[k] = result[i].reg[k] / this->scale;
		}

		for (int j = 0; j < result[i].pts.size(); j++)
		{
			result[i].pts[j].x = result[i].pts[j].x / this->scale;
			result[i].pts[j].y = result[i].pts[j].y / this->scale;
		}
		

	}
}