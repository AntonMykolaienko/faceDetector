package com.fs.facedetector;

import org.apache.commons.lang3.time.StopWatch;
import org.bytedeco.javacpp.Loader;
import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvScalar;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U;
import org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_core.cvClearMemStorage;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvGetSeqElem;
import static org.bytedeco.javacpp.opencv_core.cvGetSize;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvPoint;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_INTER_LINEAR;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.cvRectangle;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;
import org.bytedeco.javacpp.opencv_objdetect;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;

/**
 *
 * @author Anton Mykolaienko
 */
public class FaceDetector {

    private static final int SCALE = 1;
    // scaling factor to reduce size of input image
    // cascade definition for face detection
    private static final String CASCADE_FILE = "haarcascade_frontalface_alt.xml";
    //private static final String CASCADE_FILE = "haarcascade_frontalface_default.xml";
    private static final String IN_FILE = "13914093_1280734308603253_3630358304986426066_o.jpg";
    //private static final String IN_FILE = "12523060_1676564889248395_4153418241686234196_n.jpg";
    //private static final String IN_FILE = "IMG_5126.JPG";
    private static final String OUT_FILE = "markedFaces.jpg";

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // preload the opencv_objdetect module to work around a known bug
        Loader.load(opencv_objdetect.class);
        new FaceDetector().detectFaces();
    }

    private void detectFaces() {
        StopWatch sw = new StopWatch();
        try {
            sw.start();
            // load an image 
            System.out.println("Loading image from " + IN_FILE);
            IplImage origImg = loadImage();

            // convert to grayscale 
            IplImage grayImg = cvCreateImage(cvGetSize(origImg), IPL_DEPTH_8U, 1);
            cvCvtColor(origImg, grayImg, CV_BGR2GRAY);
            // scale the grayscale(to speed up face detection)
            IplImage smallImg = IplImage.create(grayImg.width() / SCALE, grayImg.height() / SCALE, IPL_DEPTH_8U, 1);
            cvResize(grayImg, smallImg, CV_INTER_LINEAR);

            // equalize the small grayscale
            //cvEqualizeHist(smallImg, smallImg);
            // create temp storage, used during object detection
            CvMemStorage storage = CvMemStorage.create();
            // instantiate a classifier cascade for face detection
            CvHaarClassifierCascade cascade = loadCascade(CASCADE_FILE);
            System.out.println("Detecting faces...");
            CvSeq faces = cvHaarDetectObjects(smallImg, cascade, storage, 1.03, 17, CV_HAAR_DO_CANNY_PRUNING);
            cvClearMemStorage(storage);

            // draw thick yellow rectangles around all the faces
            int total = faces.total();
            System.out.println("Found " + total + " face(s)");
            int filteredFaces = 0;
            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(faces, i));

                cvRectangle(origImg,
                        cvPoint(r.x() * SCALE, r.y() * SCALE),
                        cvPoint((r.x() + r.width()) * SCALE, (r.y() + r.height()) * SCALE),
                        CvScalar.GREEN, 2, CV_AA, 0);
                // undo image scaling when calculating rect coordinates
                filteredFaces++;
            }
            System.out.println("Faces after filtering: " + filteredFaces);
            System.out.println("Time taken: " + sw.toString());

            if (total > 0) {
                System.out.println("Saving marked - faces version of " + IN_FILE + " in " + OUT_FILE);
                cvSaveImage(OUT_FILE, origImg);
            }
        } finally {
            if (sw.isStarted()) {
                sw.stop();
            }
        }
    }

    private CvHaarClassifierCascade loadCascade(String fileName) {
        return new CvHaarClassifierCascade(cvLoad(getClass().getResource("/" + fileName).getPath().substring(1)));
    }

    private IplImage loadImage() {
        return cvLoadImage(getClass().getResource("/" + IN_FILE).getPath().substring(1));
    }

    private boolean isInside(CvRect parent, CvRect child) {
        return parent.x() < child.x()
                && parent.x() + parent.width() > child.x() + child.width()
                && parent.y() < child.y()
                && parent.y() + parent.height() > child.y() + child.height();
    }
}
