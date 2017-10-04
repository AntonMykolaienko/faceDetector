package com.fs.facedetector;

import org.apache.commons.lang3.time.StopWatch;
import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvScalar;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_core.cvClearMemStorage;
import static org.bytedeco.javacpp.opencv_core.cvGetSeqElem;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvPoint;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.cvRectangle;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.CascadeClassifier;

/**
 *
 * @author Anton Mykolaienko
 * @since 1.0.0
 */
public class Main {

    private static final String FACE_DETECTOR = "haarcascade_frontalface_alt.xml";

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        //OpenCV.loadShared();
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("\nRunning FaceDetector");

        new Main().run();
    }

    private void run() {
        StopWatch sw = new StopWatch();

        try {
            sw.start();

            IplImage img = cvLoadImage(getClass().getResource("/13914093_1280734308603253_3630358304986426066_o.jpg").getPath().substring(1));
            CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(
                    getClass().getResource("/" + FACE_DETECTOR).getPath().substring(1)));
            CvMemStorage storage = CvMemStorage.create();
            CvSeq sign = cvHaarDetectObjects(
                    img,
                    cascade,
                    storage,
                    1.3,
                    3,
                    CV_HAAR_DO_CANNY_PRUNING);

            cvClearMemStorage(storage);

            int totalFaces = sign.total();
            
            System.out.println(String.format("Detected %s faces", totalFaces));

            for (int i = 0; i < totalFaces; i++) {
                CvRect r = new CvRect(cvGetSeqElem(sign, i));
                cvRectangle(
                        img,
                        cvPoint(r.x(), r.y()),
                        cvPoint(r.width() + r.x(), r.height() + r.y()),
                        CvScalar.GREEN,
                        2,
                        CV_AA,
                        0);

            }
            
            System.out.println("Time: " + sw.toString());
            
            //cvShowImage("Result", img);
            cvSaveImage("output.jpg", img);
            //cvWaitKey(0);

//            CascadeClassifier faceDetector = loadCascadeClassifier();
//            Mat image = loadImage();
//            MatOfRect faceDetections = new MatOfRect();
//            //faceDetector.detectMultiScale(image, faceDetections);
//
////            Mat mGray = image;
////            Imgproc.cvtColor(image, mGray, Imgproc.COLOR_RGBA2GRAY); // Convert to grayscale
//            faceDetector.detectMultiScale(image, faceDetections, 1.1, 2, 2, new Size(40, 40), new Size(1400, 1400));
//            //faceDetector.detectMultiScale(image, faceDetections);
//
//            int faces = faceDetections.toArray().length;
//            System.out.println(String.format("Detected %s faces", faces));
//
//            if (faces > 0) {
//                for (Rect rect : faceDetections.toArray()) {
//                    Core.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
//                            new Scalar(0, 255, 0), 4);
//                }
//
//                String filename = "ouput.jpg";
//                System.out.println(String.format("Writing %s", filename));
//                Highgui.imwrite(filename, image);
//                System.out.println("Time: " + sw.toString());
//            }
        } finally {
            if (sw.isStarted()) {
                sw.stop();
            }
        }
    }

    private CascadeClassifier loadCascadeClassifier() {
        CascadeClassifier classifier = new CascadeClassifier();
        //boolean isLoaded = classifier.load(getClass().getResource("/lbpcascades/lbpcascade_frontalface.xml").getPath()
        //boolean isLoaded = classifier.load(getClass().getResource("/lbpcascade_frontalface.xml").getPath()
        boolean isLoaded = classifier.load(getClass().getResource("/" + FACE_DETECTOR).getPath()
                .substring(1));
        if (isLoaded) {
            return classifier;
        } else {
            throw new RuntimeException("Cannot load classifier");
        }
    }

    private Mat loadImage() {
        //return Highgui.imread(getClass().getResource("/13517440_1813459585550987_3047715655510740199_o.jpg").getPath().substring(1));
        return Highgui.imread(getClass().getResource("/IMG_5126.JPG").getPath().substring(1));
    }
}
