package com.mtcnn_as;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

import static android.content.ContentValues.TAG;

public class MainActivity extends Activity {

    private static final int SELECT_IMAGE1 = 1, SELECT_IMAGE2 = 2;
    private ImageView imageView1, imageView2;
    private Bitmap yourSelectedImage1 = null, yourSelectedImage2 = null;
    private Bitmap faceImage1 = null, faceImage2 = null;
    TextView faceInfo1, faceInfo2, cmpResult;   //显示face 检测的结果和compare的结果

    // 初始参数设置，可以按需修改
    private int minFaceSize = 40;
    private int testTimeCount = 1;
    private int threadsNumber = 2;

    private Face mFace = new Face();

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE" };

    public static void verifyStoragePermissions(Activity activity) {
        try {
            //检测是否有写的权限
            int permission = ActivityCompat.checkSelfPermission(activity,
                    "android.permission.WRITE_EXTERNAL_STORAGE");
            if (permission != PackageManager.PERMISSION_GRANTED) {
                // 没有写的权限，去申请写的权限，会弹出对话框
                ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE,REQUEST_EXTERNAL_STORAGE);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        verifyStoragePermissions(this);
        //拷贝模型到sk卡
        try {
            copyBigDataToSD("det1.bin");
            copyBigDataToSD("det2.bin");
            copyBigDataToSD("det3.bin");
            copyBigDataToSD("det1.param");
            copyBigDataToSD("det2.param");
            copyBigDataToSD("det3.param");
            copyBigDataToSD("liveness.bin");
            copyBigDataToSD("liveness.param");
            copyBigDataToSD("mobilefacenet.bin");
            copyBigDataToSD("mobilefacenet.param");
            copyBigDataToSD("retina.bin");
            copyBigDataToSD("retina.param");
        } catch (IOException e) {
            e.printStackTrace();
        }
        //模型初始化
        File sdDir = Environment.getExternalStorageDirectory();//获取跟目录
        String sdPath = sdDir.toString() + "/facem/";
        mFace.FaceDetectionModelInit(sdPath);

        // 多线程设置
        Log.i(TAG, "最小人脸："+minFaceSize);
        mFace.SetMinFaceSize(minFaceSize);
        mFace.SetTimeCount(testTimeCount);
        mFace.SetThreadsNumber(threadsNumber);

        //左边的图片
        imageView1 = (ImageView) findViewById(R.id.imageView1);
        faceInfo1 = (TextView)findViewById(R.id.faceInfo1);
        Button buttonImage1 = (Button) findViewById(R.id.select1);
        buttonImage1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE1);
            }
        });
        //第一张图片人脸检测
        Button buttonDetect1 = (Button) findViewById(R.id.detect1);
        buttonDetect1.setOnClickListener(new View.OnClickListener() {
             @Override
             public void onClick(View arg0) {
                 if (yourSelectedImage1 == null)
                     return;

                 //人脸检测
                 faceImage1 = null;
                 int width = yourSelectedImage1.getWidth();
                 int height = yourSelectedImage1.getHeight();
                 byte[] imageDate = getPixelsRGBA(yourSelectedImage1);

                 long timeDetectFace = System.currentTimeMillis();   //检测起始时间
                 int faceInfo[] = mFace.FaceDetect2(imageDate, width, height, 4); //只检测最大人脸，速度有较大提升
                 timeDetectFace = System.currentTimeMillis() - timeDetectFace; //人脸检测时间

                 if (faceInfo.length > 1) {       //检测到人脸

                     int faceNum = faceInfo[0];
                     Log.i(TAG, "pic width：" + width + "height：" + height + " face num：" + faceNum);
                     Bitmap drawBitmap = yourSelectedImage1.copy(Bitmap.Config.ARGB_8888, true);
                     for (int i = 0; i < faceInfo[0]; i++) {     //画出人脸识别框
                         int left, top, right, bottom;
                         Canvas canvas = new Canvas(drawBitmap);
                         Paint paint = new Paint();
                         left = faceInfo[1 + 14 * i];
                         top = faceInfo[2 + 14 * i];
                         right = faceInfo[3 + 14 * i];
                         bottom = faceInfo[4 + 14 * i];
                         paint.setColor(Color.BLUE);
                         paint.setStyle(Paint.Style.STROKE);//不填充
                         paint.setStrokeWidth(5);  //线的宽度
                         canvas.drawRect(left, top, right, bottom, paint);

                         //画特征点
                         canvas.drawPoints(new float[]{faceInfo[5+14*i],faceInfo[10+14*i],
                         faceInfo[6+14*i],faceInfo[11+14*i],
                         faceInfo[7+14*i],faceInfo[12+14*i],
                         faceInfo[8+14*i],faceInfo[13+14*i],
                         faceInfo[9+14*i],faceInfo[14+14*i]}, paint);//画多个点
                     }
                     imageView1.setImageBitmap(drawBitmap);
                     faceImage1 = Bitmap.createBitmap(yourSelectedImage1, faceInfo[1], faceInfo[2], faceInfo[3] - faceInfo[1], faceInfo[4] - faceInfo[2]);
                     Bitmap faceImage1_1 = faceImage1.copy(Bitmap.Config.ARGB_8888, true);
                     byte[] faceimage1Date = getPixelsRGBA(faceImage1_1);
                     int width1 = faceImage1.getWidth();
                     int height1 = faceImage1.getHeight();
                     long timeLiveNess = System.currentTimeMillis();   //检测起始时间
                     float scores[] = mFace.LiveNess(faceimage1Date, width1, height1, 4); //只检测最大人脸，速度有较大提升
                     timeLiveNess = System.currentTimeMillis() - timeLiveNess; //人脸检测时间
                     faceInfo1.setText("pic1 detect time:" + timeDetectFace+",liveness time:"+timeLiveNess+","+scores[0]+","+scores[1]);
                 } else {     //没有人脸
                     faceInfo1.setText("no face");
                 }
             }
        });

        //右边的图片
        imageView2 = (ImageView) findViewById(R.id.imageView2);
        faceInfo2 = (TextView) findViewById(R.id.faceInfo2);
        Button buttonImage2 = (Button) findViewById(R.id.select2);
        buttonImage2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE2);
            }
        });

        Button buttonDetect2 = (Button) findViewById(R.id.detect2);
        buttonDetect2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (yourSelectedImage2 == null)
                    return;

                //人脸检测
                faceImage2 = null;
                int width = yourSelectedImage2.getWidth();
                int height = yourSelectedImage2.getHeight();
                byte[] imageDate = getPixelsRGBA(yourSelectedImage2);

                long timeDetectFace = System.currentTimeMillis();
                int faceInfo[] = mFace.FaceDetect2(imageDate, width, height, 4);  //只检测最大人脸
                timeDetectFace = System.currentTimeMillis() - timeDetectFace;

                if (faceInfo.length > 1) {
                    //faceInfo2.setText("pic2 detect time:" + timeDetectFace);
                    int faceNum = faceInfo[0];
                    Log.i(TAG, "pic width：" + width + "height：" + height + " face num：" + faceNum);
                    Bitmap drawBitmap = yourSelectedImage2.copy(Bitmap.Config.ARGB_8888, true);

                    for (int i = 0; i < faceInfo[0]; i++) {    //画出人脸识别框
                        int left, top, right, bottom;
                        Canvas canvas = new Canvas(drawBitmap);
                        Paint paint = new Paint();
                        left = faceInfo[1 + 14 * i];
                        top = faceInfo[2 + 14 * i];
                        right = faceInfo[3 + 14 * i];
                        bottom = faceInfo[4 + 14 * i];
                        paint.setColor(Color.GREEN);
                        paint.setStyle(Paint.Style.STROKE);     //不填充
                        paint.setStrokeWidth(5);                //线的宽度
                        canvas.drawRect(left, top, right, bottom, paint);

                        //画特征点
                        canvas.drawPoints(new float[]{faceInfo[5+14*i],faceInfo[10+14*i],
                        faceInfo[6+14*i],faceInfo[11+14*i],
                        faceInfo[7+14*i],faceInfo[12+14*i],
                        faceInfo[8+14*i],faceInfo[13+14*i],
                        faceInfo[9+14*i],faceInfo[14+14*i]}, paint);//画多个点
                    }
                    imageView2.setImageBitmap(drawBitmap);
                    faceImage2 = Bitmap.createBitmap(yourSelectedImage2, faceInfo[1], faceInfo[2], faceInfo[3] - faceInfo[1], faceInfo[4] - faceInfo[2]);
                    Bitmap faceImage2_2 = faceImage2.copy(Bitmap.Config.ARGB_8888, true);
                    byte[] faceimage2Date = getPixelsRGBA(faceImage2_2);
                    int width2 = faceImage2.getWidth();
                    int height2 = faceImage2.getHeight();
                    long timeLiveNess = System.currentTimeMillis();   //检测起始时间
                    float scores[] = mFace.LiveNess(faceimage2Date, width2, height2, 4); //只检测最大人脸，速度有较大提升
                    timeLiveNess = System.currentTimeMillis() - timeLiveNess; //人脸检测时间
                    faceInfo2.setText("pic1 detect time:" + timeDetectFace+",liveness time:"+timeLiveNess+","+scores[0]+","+scores[1]);
                } else {
                    faceInfo2.setText("no face");
                }
            }
        });

        //人脸识别(compare)
        cmpResult = (TextView) findViewById(R.id.textView1);
        Button cmpImage = (Button) findViewById(R.id.facecmp);
        cmpImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (faceImage1 == null || faceImage2 == null) { //检测的人脸图片为空
                    cmpResult.setText("no enough face,return");
                    return;
                }

                byte[] faceDate1 = getPixelsRGBA(faceImage1);
                byte[] faceDate2 = getPixelsRGBA(faceImage2);
                long timeRecognizeFace = System.currentTimeMillis();
                double similar = mFace.FaceRecognize(faceDate1,faceImage1.getWidth(),faceImage1.getHeight(),
                        faceDate2,faceImage2.getWidth(),faceImage2.getHeight());
                timeRecognizeFace = System.currentTimeMillis() - timeRecognizeFace;
                cmpResult.setText("cosin:" + similar + "\n" + "cmp time:" + timeRecognizeFace);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            try {
                if (requestCode == SELECT_IMAGE1) {
                    Bitmap bitmap = decodeUri(selectedImage);

                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);

                    // resize to 227x227
                    //yourSelectedImage1 = Bitmap.createScaledBitmap(rgba, 227, 227, false);
                    yourSelectedImage1 = rgba;

                    imageView1.setImageBitmap(yourSelectedImage1);
                }
                else if (requestCode == SELECT_IMAGE2) {
                    Bitmap bitmap = decodeUri(selectedImage);
                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    yourSelectedImage2 = rgba;
                    imageView2.setImageBitmap(yourSelectedImage2);
                }
            } catch (FileNotFoundException e) {
                Log.e("MainActivity", "FileNotFoundException");
                return;
            }
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 400;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
                    || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        return BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);
    }

    //提取像素点
    private byte[] getPixelsRGBA(Bitmap image) {
        // calculate how many bytes our image consists of
        int bytes = image.getByteCount();
        ByteBuffer buffer = ByteBuffer.allocate(bytes); // Create a new buffer
        image.copyPixelsToBuffer(buffer); // Move the byte data to the buffer
        byte[] temp = buffer.array(); // Get the underlying array containing the

        return temp;
    }

    private void copyBigDataToSD(String strOutFileName) throws IOException {
        Log.i(TAG, "start copy file " + strOutFileName);
        File sdDir = Environment.getExternalStorageDirectory();//获取跟目录
        File file = new File(sdDir.toString()+"/facem/");
        if (!file.exists()) {
            file.mkdir();
        }

        String tmpFile = sdDir.toString()+"/facem/" + strOutFileName;
        File f = new File(tmpFile);
        if (f.exists()) {
            Log.i(TAG, "file exists " + strOutFileName);
            return;
        }
        InputStream myInput;
        java.io.OutputStream myOutput = new FileOutputStream(sdDir.toString()+"/facem/"+ strOutFileName);
        myInput = this.getAssets().open(strOutFileName);
        byte[] buffer = new byte[1024];
        int length = myInput.read(buffer);
        while (length > 0) {
            myOutput.write(buffer, 0, length);
            length = myInput.read(buffer);
        }
        myOutput.flush();
        myInput.close();
        myOutput.close();
        Log.i(TAG, "end copy file " + strOutFileName);
    }
    @Override
    protected void onDestroy() {
        mFace.FaceDetectionModelUnInit();
        super.onDestroy();

    }
}
