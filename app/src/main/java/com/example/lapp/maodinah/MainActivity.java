package com.example.lapp.maodinah;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
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
import org.opencv.imgproc.Imgproc;

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

    static {
        if (!OpenCVLoader.initDebug()) {
            // do something
        } else {
            // quando inicializado
            initialized = true;
        }
    }

    JavaCameraView javaCameraView;

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
            javaCameraView.setMaxFrameSize(800,600);
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

    }

    @Override
    public void onCameraViewStopped() {
        if (javaCameraView.isEnabled()) {
            javaCameraView.disableView();
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        /*Mat mRgba = inputFrame.rgba();
        Mat mRgbaT = mRgba.t();
        Core.flip(mRgba.t(), mRgbaT, 1);
        Imgproc.resize(mRgbaT, mRgbaT, mRgba.size());
        //return mRgbaT;
        */

        return paulsMethodForImageTransformation(inputFrame.rgba());
    }

    public Mat paulsMethodForImageTransformation(Mat mat) {
        Mat gray = new Mat();
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY);
        //edges = new Mat(height,width,CvType.CV_8UC1);
        //lines = new Mat(height,width,mat.type());
        Imgproc.Canny(gray, gray, limiarInferior,limiarSuperior);
        //Imgproc.CV
        return gray;
        /*List<MatOfPoint> contornos = new ArrayList<MatOfPoint>();
        Mat h = new Mat();

        Imgproc.findContours(gray, contornos, h, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contornos.size(); i++) {
            Imgproc.drawContours(mat, contornos, i, new Scalar(0, 0, 255), -1);
        }

        return mat;*/

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
