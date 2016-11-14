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
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.core.TermCriteria;

import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Collection;
import java.util.Vector;


import static android.R.attr.bitmap;
import static android.R.attr.max;

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
    private Mat nonZero = new Mat();
    private Mat MatrizTocada;
    private Mat roiHist;
    private Scalar minHSV = new Scalar(3);
    private Scalar maxHSV = new Scalar(3);
    private List<Point> dedos;
    private TermCriteria condicaopraparar = new TermCriteria(TermCriteria.COUNT | TermCriteria.EPS, 10, 1);

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
    private Mat framesegundo;
    private Mat h = new Mat(); // hierarquia findContours
    private List<MatOfPoint> contornos;
    private Scalar lowerBound;
    private Scalar upperBound;
    private float offsetFactX, offsetFactY;
    private float scaleFactX, scaleFactY;
    private MatOfPoint hullPoints;
    private MatOfInt hull;
    private MatOfInt channels = new MatOfInt(0);
    private MatOfFloat hue_range = new MatOfFloat(0, 180);
    private Mat dstBProject = new Mat();
    private List<Mat> allRoiHis = new ArrayList<>();
    private Mat gray = new Mat();
    private Mat trash = new Mat();
    private Mat destino = new Mat();
    private Mat cannyDestino = new Mat();

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
        hullPoints = new MatOfPoint();
        hull = new MatOfInt();
    }

    @Override
    public void onCameraViewStopped() {
        frame.release();

        if (javaCameraView.isEnabled()) {
            javaCameraView.disableView();
        }
    }


    private void metodoEscroto(Mat frame){

        int x = (int)centroid.x;
        int y = (int)centroid.y;

        int rows = frame.rows();
        int cols = frame.cols();

        Rect retanguloTocado = new Rect();
        int ladoRet = 30;

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

            tocou = !tocou;
        }

        return false;
    }

    private ArrayList<MatOfPoint> getAllContornos(Mat frame){
        framesegundo = frame.clone();
        ArrayList<MatOfPoint> contor = new ArrayList<MatOfPoint>();
        Imgproc.findContours(framesegundo,
                contor,
                h,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE );

        return contor;
    }

    protected int getPalmContour(List<MatOfPoint> contours){

        Rect roi;
        int indexOfMaxContour = -1;
        for (int i = 0; i < contours.size(); i++) {
            roi = Imgproc.boundingRect(contours.get(i));
            if(roi.contains(centroid))
                return i;
        }
        return indexOfMaxContour;
    }

    protected Point getDistanceTransformCenter(Mat frame){

        Imgproc.distanceTransform(frame, frame, Imgproc.CV_DIST_L2, 3);
        frame.convertTo(frame, CvType.CV_8UC1);
        Core.normalize(frame, frame, 0, 255, Core.NORM_MINMAX);
        Imgproc.threshold(frame, frame, 254, 255, Imgproc.THRESH_TOZERO);
        Core.findNonZero(frame, nonZero);

        // have to manually loop through matrix to calculate sums
        int sumx = 0, sumy = 0;
        for(int i=0; i<nonZero.rows(); i++) {
            sumx += nonZero.get(i, 0)[0];
            sumy += nonZero.get(i, 0)[1];
        }
        sumx /= nonZero.rows();
        sumy /= nonZero.rows();

        return new Point(sumx, sumy);
    }
    protected List<Point> getConvexHullPoints(MatOfPoint contour){
        Imgproc.convexHull(contour, hull);
        List<Point> hullPoints = new ArrayList<>();
        for(int j=0; j < hull.toList().size(); j++){
            hullPoints.add(contour.toList().get(hull.toList().get(j)));
        }
        return hullPoints;
    }

    protected double getEuclDistance(Point one, Point two){
        return Math.sqrt(Math.pow((two.x - one.x), 2)
                + Math.pow((two.y - one.y), 2));
    }
    protected List<Point> getDedos(List<Point> hullPoints, int rows){
        double betwFingersThresh = 80;
        double distFromCenterThresh = 80;
        double thresh = 80;
        List<Point> fingerTips  = new ArrayList<>();
        for(int i=0; i<hullPoints.size(); i++){
            Point point = hullPoints.get(i);
            if(rows - point.y < thresh)
                continue;
            if(fingerTips.size() == 0){
                fingerTips.add(point);
                continue;
            }
            Point prev = fingerTips.get(fingerTips.size() - 1);
            double euclDist = getEuclDistance(prev, point);

            if(getEuclDistance(prev, point) > thresh/2 &&
                    getEuclDistance(centroid, point) > thresh)
                fingerTips.add(point);

            if(fingerTips.size() == 5)
                break;
        }
        return fingerTips;
    }

    private void assignRoiHist(Point ponto, Mat frame, Mat roiHist, Rect roi){
        int ladoMetade = 15;
        if ((ponto.x - ladoMetade) > 0) {
            roi.x = ((int)ponto.x - ladoMetade);
        } else {
            roi.x = 0;
        }
        if ((ponto.y - ladoMetade) > 0) {
            roi.y = (int)(ponto.y - ladoMetade);
        } else {
            roi.y = 0;
        }

        if (2*ladoMetade > frame.width()) {
            roi.width = 2*ladoMetade;
        } else {
            roi.width = frame.width();
        }

        if (2*ladoMetade > frame.height()) {
            roi.height = 2*ladoMetade;
        } else {
            roi.height = frame.height();
        }

        Mat subMatriz = frame.submat(roi);
        Mat mask = new Mat();
        MatOfInt histogramaSize = new MatOfInt(180);

        Imgproc.cvtColor(subMatriz,subMatriz,Imgproc.COLOR_RGB2HSV_FULL);
        Core.inRange(subMatriz,minHSV,maxHSV,mask);
        List<Mat> MatAuxiliarLista = new ArrayList<>();
        MatAuxiliarLista.add(subMatriz);
        Imgproc.calcHist(MatAuxiliarLista, channels, mask, roiHist, histogramaSize, hue_range);
        Core.normalize(roiHist, roiHist, 0, 255, Core.NORM_MINMAX);
    }

    /*
    private Rect motionTrack(Mat frame, Rect roi,Mat roihist){
        //Metodo Legal
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2HSV_FULL);
        List<Mat> auxLista = new ArrayList<>();
        auxLista.add(frame);
        Imgproc.calcBackProject(auxLista, channels, allRoiHis.get(0) , dstBProject, hue_range , 1);

        Video.meanShift(dstBProject, roi,condicaopraparar);
        return roi;
    }
    */

    private Mat linhaMao(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        Mat l1 = inputFrame.rgba();
        Mat l2 = l1.clone();





        return l1;
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

    public Mat paulsMethodForImageTransformation(Mat inputFrame) {

        frame = inputFrame.clone();

        Imgproc.cvtColor(frame,destino,Imgproc.COLOR_BGRA2GRAY);

        Imgproc.Canny(destino, destino, 100, 255);

       // Vector<Vec2f>1


        //Imgproc.threshold(frame,trash, 50, 255,Imgproc.THRESH_BINARY);


        if  (tocou){

            frame = inputFrame.clone();
            Imgproc.GaussianBlur(frame, frame, new Size(9, 9), 5);
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2HSV_FULL);
            Core.inRange(frame, minHSV, maxHSV, frame);
            //Imgproc.threshold(frame, frame, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
            contornos = getAllContornos(frame);
            int indiceContorno = getPalmContour(contornos);

            if  (indiceContorno == -1) {
                return frame;
            }
            else {
                Point palma = getDistanceTransformCenter(frame);
                Rect roi = Imgproc.boundingRect(contornos.get(indiceContorno));

                List<Point> hullPoints = getConvexHullPoints(contornos.get(indiceContorno));
                dedos = getDedos(hullPoints, frame.rows());
                Collections.reverse(dedos);
                for(int i = 0; i+1 < dedos.size(); i++){
                    Imgproc.line(frame, dedos.get(i), dedos.get(i+1), new Scalar(255,255,255),2);
                }

                Imgproc.drawContours(frame, contornos, -1, new Scalar(200, 200, 0), 2);

                return frame;
            }
        } else return destino;



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
