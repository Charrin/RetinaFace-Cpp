#ifndef ANCHOR_GENERTOR
#define ANCHOR_GENERTOR

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <time.h>
#include <map>
#include <math.h>

#include "config.h"
//#include "opencv2/opencv.hpp"

namespace my
{

    struct Size
    {
        Size() : width(0), height(0) {}
        Size(int _w, int _h) : width(_w), height(_h) {}

        int width;
        int height;
    };

    template<typename _Tp>
    struct Rect_
    {
        Rect_() : x(0), y(0), width(0), height(0) {}
        Rect_(_Tp _x, _Tp _y, _Tp _w, _Tp _h) : x(_x), y(_y), width(_w), height(_h) {}

        _Tp x;
        _Tp y;
        _Tp width;
        _Tp height;

        // area
        _Tp area() const
        {
            return width * height;
        }
    };

    template<typename _Tp> static inline Rect_<_Tp>& operator &= (Rect_<_Tp>& a, const Rect_<_Tp>& b)
    {
        _Tp x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
        a.width = std::min(a.x + a.width, b.x + b.width) - x1;
        a.height = std::min(a.y + a.height, b.y + b.height) - y1;
        a.x = x1; a.y = y1;
        if (a.width <= 0 || a.height <= 0)
            a = Rect_<_Tp>();
        return a;
    }

    template<typename _Tp> static inline Rect_<_Tp>& operator |= (Rect_<_Tp>& a, const Rect_<_Tp>& b)
    {
        _Tp x1 = std::min(a.x, b.x), y1 = std::min(a.y, b.y);
        a.width = std::max(a.x + a.width, b.x + b.width) - x1;
        a.height = std::max(a.y + a.height, b.y + b.height) - y1;
        a.x = x1; a.y = y1;
        return a;
    }

    template<typename _Tp> static inline Rect_<_Tp> operator & (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
    {
        Rect_<_Tp> c = a;
        return c &= b;
    }

    template<typename _Tp> static inline Rect_<_Tp> operator | (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
    {
        Rect_<_Tp> c = a;
        return c |= b;
    }

    typedef Rect_<int> Rect;
    typedef Rect_<float> Rect2f;
    typedef Rect_<double> Rect2d;

    template<typename _Tp>
    struct Point_
    {
        Point_() : x(0), y(0) {}
        Point_(_Tp _x, _Tp _y) : x(_x), y(_y) {}

        _Tp x;
        _Tp y;
    };

    typedef Point_<int> Point;
    typedef Point_<float> Point2f;
}

using namespace my;

class CRect2f {
public:
    CRect2f(float x1, float y1, float x2, float y2) {
        val[0] = x1;
        val[1] = y1;
        val[2] = x2;
        val[3] = y2;
    }

    float& operator[](int i) {
        return val[i];
    }

    float operator[](int i) const {
        return val[i];
    }

    float val[4];

    void print() {
        printf("rect %f %f %f %f\n", val[0], val[1], val[2], val[3]);
    }
};

class Anchor {
public:
	Anchor() {
	}

	~Anchor() {
	}

    bool operator<(const Anchor &t) const {
        return score < t.score;
    }

    bool operator>(const Anchor &t) const {
        return score > t.score;
    }

    float& operator[](int i) {
        //assert(0 <= i && i <= 4);

        if (i == 0) 
            return finalbox.x;
        if (i == 1) 
            return finalbox.y;
        if (i == 2) 
            return finalbox.width;
        if (i == 3) 
            return finalbox.height;
    }

    float operator[](int i) const {
        //assert(0 <= i && i <= 4);

        if (i == 0) 
            return finalbox.x;
        if (i == 1) 
            return finalbox.y;
        if (i == 2) 
            return finalbox.width;
        if (i == 3) 
            return finalbox.height;
    }

    Rect2f anchor; // x1,y1,x2,y2
	float reg[4]; // offset reg
    Point center; // anchor feat center
	float score; // cls score
    std::vector<Point2f> pts; // pred pts

    Rect2f finalbox; // final box res

    void print() {
        printf("finalbox %f %f %f %f, score %f\n", finalbox.x, finalbox.y, finalbox.width, finalbox.height, score);
        printf("landmarks ");
        for (int i = 0; i < pts.size(); ++i) {
            printf("%f %f, ", pts[i].x, pts[i].y);
        }
        printf("\n");
    }
};

class AnchorGenerator {
public:
	AnchorGenerator();
	~AnchorGenerator();

    // init different anchors
    int Init(int stride, const AnchorCfg& cfg, bool dense_anchor);

    // anchor plane
    int Generate(int fwidth, int fheight, int stride, float step, std::vector<int>& size, std::vector<float>& ratio, bool dense_anchor);

	// filter anchors and return valid anchors
	int FilterAnchor(ncnn::Mat& cls, ncnn::Mat& reg, ncnn::Mat& pts, std::vector<Anchor>& result);

private:
    void _ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios, std::vector<CRect2f>& ratio_anchors);

    void _scale_enum(const std::vector<CRect2f>& ratio_anchor, const std::vector<float>& scales, std::vector<CRect2f>& scale_anchors);

    void bbox_pred(const CRect2f& anchor, const CRect2f& delta, Rect2f& box);

    void landmark_pred(const CRect2f anchor, const std::vector<Point2f>& delta, std::vector<Point2f>& pts);

	std::vector<std::vector<Anchor>> anchor_planes; // corrspont to channels

	std::vector<int> anchor_size; 
	std::vector<float> anchor_ratio;
	float anchor_step; // scale step
	int anchor_stride; // anchor tile stride
	int feature_w; // feature map width
	int feature_h; // feature map height

    std::vector<CRect2f> preset_anchors;
	int anchor_num; // anchor type num

};

#endif // ANCHOR_GENERTOR
