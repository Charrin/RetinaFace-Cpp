#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include <cstring>

// ncnn
#include "net.h"
#include "recognize.h"
#include "retinaface.h"

using namespace Face;

#define TAG "MtcnnSo"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)
static Recognize *mRecognize;
static RetinaFace *mRetinaFace;

//sdk是否初始化成功
bool detection_sdk_init_ok = false;


extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_mtcnn_1as_Face_FaceDetectionModelInit(JNIEnv *env, jobject instance,
                                                jstring faceDetectionModelPath_) {
     LOGD("JNI开始人脸检测模型初始化");
    //如果已初始化则直接返回
    if (detection_sdk_init_ok) {
        //  LOGD("人脸检测模型已经导入");
        return true;
    }
    jboolean tRet = false;
    if (NULL == faceDetectionModelPath_) {
        //   LOGD("导入的人脸检测的目录为空");
        return tRet;
    }

    //获取MTCNN模型的绝对路径的目录（不是/aaa/bbb.bin这样的路径，是/aaa/)
    const char *faceDetectionModelPath = env->GetStringUTFChars(faceDetectionModelPath_, 0);
    if (NULL == faceDetectionModelPath) {
        return tRet;
    }

    string tFaceModelDir = faceDetectionModelPath;
    string tLastChar = tFaceModelDir.substr(tFaceModelDir.length() - 1, 1);
    //LOGD("init, tFaceModelDir last =%s", tLastChar.c_str());
    //目录补齐/
    if ("\\" == tLastChar) {
        tFaceModelDir = tFaceModelDir.substr(0, tFaceModelDir.length() - 1) + "/";
    } else if (tLastChar != "/") {
        tFaceModelDir += "/";
    }
    LOGD("init, tFaceModelDir=%s", tFaceModelDir.c_str());

    //没判断是否正确导入，懒得改了
   
    mRecognize = new Recognize(tFaceModelDir);
   
    mRetinaFace = new RetinaFace(tFaceModelDir);
    mDetect->SetMinFace(40);
    mDetect->SetNumThreads(2);    // 2线程
    mRecognize->SetThreadNum(2);
    mLiveness->SetThreadNum(2);
    env->ReleaseStringUTFChars(faceDetectionModelPath_, faceDetectionModelPath);
    detection_sdk_init_ok = true;
    tRet = true;
    return tRet;
}


JNIEXPORT jintArray JNICALL
Java_com_mtcnn_1as_Face_FaceDetect(JNIEnv *env, jobject instance, jbyteArray imageDate_,
                                 jint imageWidth, jint imageHeight, jint imageChannel) {
    //  LOGD("JNI开始检测人脸");
    if(!detection_sdk_init_ok){
        LOGD("人脸检测MTCNN模型SDK未初始化，直接返回空");
        return NULL;
    }

    int tImageDateLen = env->GetArrayLength(imageDate_);
    if(imageChannel == tImageDateLen / imageWidth / imageHeight){
        LOGD("数据宽=%d,高=%d,通道=%d",imageWidth,imageHeight,imageChannel);
    }
    else{
        LOGD("数据长宽高通道不匹配，直接返回空");
        return NULL;
    }

    jbyte *imageDate = env->GetByteArrayElements(imageDate_, NULL);
    if (NULL == imageDate){
        LOGD("导入数据为空，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    if(imageWidth<20||imageHeight<20){
        LOGD("导入数据的宽和高小于20，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    //TODO 通道需测试
    if(3 == imageChannel || 4 == imageChannel){
        //图像通道数只能是3或4；
    }else{
        LOGD("图像通道数只能是3或4，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    //int32_t minFaceSize=40;
    //mtcnn->SetMinFace(minFaceSize);

    unsigned char *faceImageCharDate = (unsigned char*)imageDate;
    ncnn::Mat ncnn_img;
    if(imageChannel==3) {
        ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_BGR2RGB,
                                          imageWidth, imageHeight);
    }else{
        ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_RGBA2RGB, imageWidth, imageHeight);
    }

    std::vector<Anchor> finalBbox;
    mRetinaFace->detect(ncnn_img, finalBbox);

    int32_t num_face = static_cast<int32_t>(finalBbox.size());
    LOGD("检测到的人脸数目：%d\n", num_face);

    int out_size = 1+num_face*14;
    LOGD("%d",out_size);
    int *faceInfo = new int[out_size];
    LOGD("内部人脸检测完成,开始导出数据");
    faceInfo[0] = num_face;
    for(int i=0;i<num_face;i++){
        faceInfo[14*i+1] = int(finalBbox[i].finalbox.x/mRetinaFace->scale);//left
        LOGD("%d",faceInfo[14*i+1]);
        faceInfo[14*i+2] = int(finalBbox[i].finalbox.y/mRetinaFace->scale);//top
        LOGD("%d",faceInfo[14*i+2]);
        faceInfo[14*i+3] = int(finalBbox[i].finalbox.width/mRetinaFace->scale);//right
        LOGD("%d",faceInfo[14*i+3]);
        faceInfo[14*i+4] = int(finalBbox[i].finalbox.height/mRetinaFace->scale);//bottom
        LOGD("%d",faceInfo[14*i+4]);
        for (int j =0;j<finalBbox[i].pts.size();j++){
            faceInfo[14*i+5+j]=int(finalBbox[i].pts[j].x/mRetinaFace->scale);
            faceInfo[14*i+10+j]=int(finalBbox[i].pts[j].y/mRetinaFace->scale);
        }
    }

    jintArray tFaceInfo = env->NewIntArray(out_size);
    env->SetIntArrayRegion(tFaceInfo,0,out_size,faceInfo);
    LOGD("内部人脸检测完成,导出数据成功");
    delete[] faceInfo;
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
    return tFaceInfo;
}


JNIEXPORT jboolean JNICALL
Java_com_mtcnn_1as_Face_FaceDetectionModelUnInit(JNIEnv *env, jobject instance) {
    if(!detection_sdk_init_ok){
        LOGD("人脸检测MTCNN模型已经释放过或者未初始化");
        return true;
    }
    jboolean tDetectionUnInit = false;
    delete mRecognize;
    delete mRetinaFace;

    detection_sdk_init_ok=false;
    tDetectionUnInit = true;
    LOGD("人脸检测初始化锁，重新置零");
    return tDetectionUnInit;

}

}


extern "C"
JNIEXPORT jdouble JNICALL
Java_com_mtcnn_1as_Face_FaceRecognize(JNIEnv *env, jobject instance,
                                       jbyteArray faceDate1_, jint w1, jint h1,
                                       jbyteArray faceDate2_, jint w2, jint h2) {
    jbyte *faceDate1 = env->GetByteArrayElements(faceDate1_, NULL);
    jbyte *faceDate2 = env->GetByteArrayElements(faceDate2_, NULL);

    // TODO
    double similar = 0;
    unsigned char *faceImageCharDate1 = (unsigned char *) faceDate1;
    unsigned char *faceImageCharDate2 = (unsigned char *) faceDate2;

    //没进行对齐操作，且以下对图像缩放的操作方法对结果影响较大。可改进空间很大，有能力的自己改改
    ncnn::Mat ncnn_img1 = ncnn::Mat::from_pixels_resize(faceImageCharDate1,
                                                        ncnn::Mat::PIXEL_RGBA2RGB, w1, h1, 112,
                                                        112);
    ncnn::Mat ncnn_img2 = ncnn::Mat::from_pixels_resize(faceImageCharDate2,
                                                        ncnn::Mat::PIXEL_RGBA2RGB, w2, h2, 112,
                                                        112);
    std::vector<float> feature1, feature2;
    mRecognize->start(ncnn_img1, feature1);
    mRecognize->start(ncnn_img2, feature2);

    env->ReleaseByteArrayElements(faceDate1_, faceDate1, 0);
    env->ReleaseByteArrayElements(faceDate2_, faceDate2, 0);
    similar = calculSimilar(feature1, feature2);
    return similar;
}
