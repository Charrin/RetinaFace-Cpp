package com.mtcnn_as;

/**
 * Created by hasee on 2017/12/19.
 */

public class Face {
    //人脸检测模型导入
    public native boolean FaceDetectionModelInit(String faceDetectionModelPath);

  

    public native int[] FaceDetect(byte[] imageDate, int imageWidth , int imageHeight, int imageChannel);


    //人脸检测模型反初始化
    public native boolean FaceDetectionModelUnInit();

    //人脸识别
    public native double FaceRecognize(byte[] faceDate1,int w1,int h1,byte[] faceDate2,int w2,int h2);

    static {
        System.loadLibrary("Face");
    }
}
