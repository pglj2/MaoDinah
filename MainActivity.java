package com.example.lapp.maodinah;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.Window;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.BackgroundSubtractorMOG2;

import java.util.ArrayList;
import java.util.List;

import static android.R.attr.bitmap;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {


    static boolean initialized;
    static int height;
    static int width;
    static double limiarSuperior = 90;
    static double limiarInferior = 45;
    static Mat edges;
    static Mat lines;
    static int threshold = 50;
    static double tamanhoMinimoLinha = 20;
    static double lineGap = 20;
    static double[] vec;
    static double x1, x2, y1, y2;
    static Point start, end;
    static Mat matGlobal;
    private boolean tocou = false;
    private Point centroid;
    private Mat MatrizTocada;
    private Scalar minHSV = new Scalar(3);
    private Scalar maxHSV = new Scalar(3);

    static {
        if (!OpenCVLoader.initDebug()) {
            // do something
        } else {
            // quando inicializado
            initialized = true;
        }
    }

    JavaCameraView javaCameraView;
    private Mat frame;
    private Mat h; // hierarquia findContours
    private List<MatOfPoint> contornos;
    private Scalar lowerBound;
    private Scalar upperBound;
    private float offsetFactX, offsetFactY;
    private float scaleFactX, scaleFactY;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportActionBar().hide();
        setContentView(R.layout.activity_main);

        // seta a view da camera
        javaCameraView = (JavaCameraView) findViewById(R.id.camera_view);

        if (initialized) {
            javaCameraView.setCameraIndex(0); // 0 = camera traseira
            javaCameraView.setCvCameraViewListener(this);
            javaCameraView.enableView();
            javaCameraView.enableFpsMeter();
            height = javaCameraView.getHeight();
            width = javaCameraView.getWidth();
            javaCameraView.setMaxFrameSize(960,480);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (javaCameraView.isEnabled()) {
            javaCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        lowerBound = new Scalar(3);
        upperBound = new Scalar(3);
        centroid  = new Point(-1,-1);
        setScaleFactors(width,height);
    }

    @Override
    public void onCameraViewStopped() {
        frame.release();

        if (javaCameraView.isEnabled()) {
            javaCameraView.disableView();
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        matGlobal = inputFrame.rgba();
        /*Mat mRgba = inputFrame.rgba();
        Mat mRgbaT = mRgba.t();
        Core.flip(mRgba.t(), mRgbaT, 1);
        Imgproc.resize(mRgbaT, mRgbaT, mRgba.size());
        //return mRgbaT;
        */
        return paulsMethodForImageTransformation(inputFrame.rgba());
    }

    private void metodoEscroto(Mat frame){

        int x = (int)centroid.x;
        int y = (int)centroid.y;

        int rows = frame.rows();
        int cols = frame.cols();

        Rect retanguloTocado = new Rect();
        int ladoRet = 20;

        if (x > ladoRet){
            retanguloTocado.x = x - ladoRet;
        }
        else {
            retanguloTocado.x =  0;
        }

        if (y > ladoRet){
            retanguloTocado.y = y - ladoRet;
        }
        else {
            retanguloTocado.y =  0;
        }

        retanguloTocado.width = (x + ladoRet < cols) ? (x + ladoRet - retanguloTocado.x) : (cols - retanguloTocado.x);
        retanguloTocado.height = (y + ladoRet < rows) ? (y + ladoRet - retanguloTocado.y) : (rows - retanguloTocado.y);

        MatrizTocada = frame.submat(retanguloTocado);

        Imgproc.cvtColor(MatrizTocada, MatrizTocada, Imgproc.COLOR_RGB2HSV_FULL);
        Scalar somatorioCores = Core.sumElems(MatrizTocada);
        int total = retanguloTocado.width * (retanguloTocado.height);
        double avgHSV[] = {somatorioCores.val[0] / total, somatorioCores.val[1] / total, somatorioCores.val[2] / total};
        assignHSV(avgHSV);
    }

    private void assignHSV(double avgHSV[]){
        //B
        minHSV.val[0] = (avgHSV[0] > 10) ? avgHSV[0] - 10 : 0;
        maxHSV.val[0] = (avgHSV[0] < 245) ? avgHSV[0] + 10 : 255;
        //G
        minHSV.val[1] = (avgHSV[1] > 130) ? avgHSV[1] - 100 : 30;
        maxHSV.val[1] = (avgHSV[1] < 155) ? avgHSV[1] + 100 : 255;
        //R
        minHSV.val[2] = (avgHSV[2] > 130) ? avgHSV[2] - 100 : 30;
        maxHSV.val[2] = (avgHSV[2] < 155) ? avgHSV[2] + 100 : 255;
    }


    protected void setScaleFactors(int vidWidth, int vidHeight){
        float deviceWidth = javaCameraView.getWidth();
        float deviceHeight = javaCameraView.getHeight();
        if(deviceHeight - vidHeight < deviceWidth - vidWidth){
            float temp = vidWidth * deviceHeight / vidHeight;
            offsetFactY = 0;
            offsetFactX = (deviceWidth - temp) / 2;
            scaleFactY = vidHeight / deviceHeight;
            scaleFactX = vidWidth / temp;
        }
        else{
            float temp = vidHeight * deviceWidth / vidWidth;
            offsetFactX= 0;
            offsetFactY = (deviceHeight - temp) / 2;
            scaleFactX = vidWidth / deviceWidth;
            scaleFactY = vidHeight / temp;
        }
    }

    public boolean onTouchEvent(MotionEvent event){

        if(!tocou){
            frame = matGlobal.clone();
            Imgproc.GaussianBlur(frame, frame , new Size(9,9),5 );

            /*
            int x = Math.round((event.getX()));
            int y = Math.round((event.getY()));
            */
            int x = Math.round((event.getX() - offsetFactX) * scaleFactX) ;
            int y = Math.round((event.getY() - offsetFactY) * scaleFactY);

            int rows = frame.rows();
            int cols = frame.cols();


            if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

            centroid.x = x;
            centroid.y = y;

            metodoEscroto(frame);

            tocou = true;
        }

        return false;
    }

    public Mat paulsMethodForImageTransformation(Mat inputFrame) {
        if  (tocou){
            frame = inputFrame.clone();
            Imgproc.GaussianBlur(frame, frame, new Size(9, 9), 5);
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2HSV_FULL);
            Core.inRange(frame, minHSV, maxHSV, frame);
            //Imgproc.threshold(frame, frame, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            return frame;
        } else return inputFrame;



        //Imgproc.adaptiveThreshold(frame, frame, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 7, 7);
        //Core.inRange(frame, lowerBound, upperBound, frame);

        /*contornos = new ArrayList<>();
        h = new Mat();
        Imgproc.findContours(frame, contornos, h, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        Rect roi;
        for (int i = 0; i < contornos.size(); i++) {
            //roi = Imgproc.boundingRect(contornos.get(i));
            //Imgproc.rectangle(inputFrame, new Point(roi.x, roi.y), new Point(roi.x+roi.width, roi.y+roi.height), new Scalar(255,0,0,255), 1, 8, 0);
            Imgproc.drawContours(inputFrame, contornos, i, new Scalar(0, 0, 255), 5);
        }*/
        //return frame;


        //edges = new Mat(height,width,CvType.CV_8UC1);
        //lines = new Mat(height,width,mat.type());
        //Imgproc.Canny(gray, gray, limiarInferior,limiarSuperior);

        /*Imgproc.HoughLinesP(
                gray,
                lines,
                1,
                Math.PI/180,
                threshold
        );

        for (int x = 0; x < lines.cols(); x++) {
            vec = lines.get(0, x);
            x1 = vec[0];
            y1 = vec[1];
            x2 = vec[2];
            y2 = vec[3];
            start = new Point(x1, y1);
            end = new Point(x2, y2);
            Imgproc.line(gray, start, end, new Scalar(255, 255, 255), 30);
        }
        return gray;*/

    }
}
