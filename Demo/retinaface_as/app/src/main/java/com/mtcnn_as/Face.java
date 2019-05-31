package com.mtcnn_as;

/**
 * Created by hasee on 2017/12/19.
 */

public class Face {
    //人脸检测模型导入
    public native boolean FaceDetectionModelInit(String faceDetectionModelPath);

    //人脸检测
    public native int[] FaceDetect(byte[] imageDate, int imageWidth , int imageHeight, int imageChannel);

    public native int[] FaceDetect2(byte[] imageDate, int imageWidth , int imageHeight, int imageChannel);

    public native int[] MaxFaceDetect(byte[] imageDate, int imageWidth , int imageHeight, int imageChannel);

    public native float[] LiveNess(byte[] imageDate, int imageWidth , int imageHeight, int imageChannel);

    //人脸检测模型反初始化
    public native boolean FaceDetectionModelUnInit();

    //检测的最小人脸设置
    public native boolean SetMinFaceSize(int minSize);

    //线程设置
    public native boolean SetThreadsNumber(int threadsNumber);

    //循环测试次数
    public native boolean SetTimeCount(int timeCount);

    //人脸识别
    public native double FaceRecognize(byte[] faceDate1,int w1,int h1,byte[] faceDate2,int w2,int h2);

    static {
        System.loadLibrary("Face");
    }
}
