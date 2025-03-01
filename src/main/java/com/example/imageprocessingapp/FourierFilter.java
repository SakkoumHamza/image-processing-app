package com.example.imageprocessingapp;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class FourierFilter {

    // Convert JavaFX Image to OpenCV Mat
    private static Mat convertImageToMat(Image image) {
        BufferedImage bufferedImage = SwingFXUtils.fromFXImage(image, null);
        Mat mat = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(), CvType.CV_8UC3);
        byte[] data = ((java.awt.image.DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
        mat.put(0, 0, data);
        return mat;
    }

    // Convert OpenCV Mat to JavaFX Image
    private static Image convertMatToImage(Mat mat) {
        Mat temp = new Mat();
        if (mat.channels() == 1) {
            Imgproc.cvtColor(mat, temp, Imgproc.COLOR_GRAY2BGR);
        } else {
            mat.copyTo(temp);
        }
        BufferedImage bufferedImage = new BufferedImage(temp.cols(), temp.rows(), BufferedImage.TYPE_3BYTE_BGR);
        temp.get(0, 0, ((java.awt.image.DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData());
        return SwingFXUtils.toFXImage(bufferedImage, null);
    }

    public static ImageView applyIdealLowPassFilter(ImageView imageView, double D0) {
        // Convert JavaFX Image to OpenCV Mat
        Mat src = convertImageToMat(imageView.getImage());

        // Convert to grayscale
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        // Perform Fourier Transform
        Mat padded = new Mat();
        int m = Core.getOptimalDFTSize(src.rows());
        int n = Core.getOptimalDFTSize(src.cols());
        Core.copyMakeBorder(src, padded, 0, m - src.rows(), 0, n - src.cols(), Core.BORDER_CONSTANT, Scalar.all(0));

        List<Mat> planes = new ArrayList<>();
        padded.convertTo(padded, CvType.CV_32F);
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        Mat complexImage = new Mat();
        Core.merge(planes, complexImage);

        Core.dft(complexImage, complexImage);

        // Create ideal low-pass filter mask
        Mat mask = createIdealLowPassFilter(complexImage.size(), D0);

        // Apply the mask
        Core.split(complexImage, planes);
        Core.multiply(planes.get(0), mask, planes.get(0));
        Core.multiply(planes.get(1), mask, planes.get(1));
        Core.merge(planes, complexImage);

        // Perform inverse Fourier Transform
        Core.idft(complexImage, complexImage, Core.DFT_SCALE | Core.DFT_REAL_OUTPUT, 0);
        // Convert back to an image
        Mat result = new Mat();
        Core.split(complexImage, planes);
        Core.normalize(planes.get(0), result, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);

        // Set the filtered image to the ImageView
        imageView.setImage(convertMatToImage(result));
        return imageView;
    }

    private static Mat createIdealLowPassFilter(Size size, double D0) {
        Mat mask = Mat.zeros(size, CvType.CV_32F);
        Point center = new Point(size.width / 2, size.height / 2);

        for (int i = 0; i < size.height; i++) {
            for (int j = 0; j < size.width; j++) {
                double d = Math.sqrt(Math.pow(i - center.y, 2) + Math.pow(j - center.x, 2));
                if (d <= D0) {
                    mask.put(i, j, 1);
                }
            }
        }

        return mask;
    }

    public static ImageView applyIdealHighPassFilter(ImageView imageView, double D0) {
        // Convert JavaFX Image to OpenCV Mat
        Mat src = convertImageToMat(imageView.getImage());

        // Convert to grayscale
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        // Perform Fourier Transform
        Mat padded = new Mat();
        int m = Core.getOptimalDFTSize(src.rows());
        int n = Core.getOptimalDFTSize(src.cols());
        Core.copyMakeBorder(src, padded, 0, m - src.rows(), 0, n - src.cols(), Core.BORDER_CONSTANT, Scalar.all(0));

        List<Mat> planes = new ArrayList<>();
        padded.convertTo(padded, CvType.CV_32F);
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        Mat complexImage = new Mat();
        Core.merge(planes, complexImage);

        Core.dft(complexImage, complexImage);

        // Create ideal high-pass filter mask
        Mat mask = createIdealHighPassFilter(complexImage.size(), D0);

        // Apply the mask
        Core.split(complexImage, planes);
        Core.multiply(planes.get(0), mask, planes.get(0));
        Core.multiply(planes.get(1), mask, planes.get(1));
        Core.merge(planes, complexImage);

        // Perform inverse Fourier Transform
        Core.idft(complexImage, complexImage, Core.DFT_SCALE | Core.DFT_REAL_OUTPUT, 0);

        // Convert back to an image
        Mat result = new Mat();
        Core.split(complexImage, planes);
        Core.normalize(planes.get(0), result, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);

        // Set the filtered image to the ImageView
        imageView.setImage(convertMatToImage(result));
        return imageView;
    }

    private static Mat createIdealHighPassFilter(Size size, double D0) {
        Mat mask = Mat.ones(size, CvType.CV_32F);
        Point center = new Point(size.width / 2, size.height / 2);

        for (int i = 0; i < size.height; i++) {
            for (int j = 0; j < size.width; j++) {
                double d = Math.sqrt(Math.pow(i - center.y, 2) + Math.pow(j - center.x, 2));
                if (d <= D0) {
                    mask.put(i, j, 0);
                }
            }
        }

        return mask;
    }
    public static ImageView applyLowButterworthFilter(ImageView imageView, double D0) {
        // Convert JavaFX Image to OpenCV Mat
        Mat src = convertImageToMat(imageView.getImage());

        // Convert to grayscale
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        // Perform Fourier Transform
        Mat padded = new Mat();
        int m = Core.getOptimalDFTSize(src.rows());
        int n = 3;
        Core.copyMakeBorder(src, padded, 0, m - src.rows(), 0, n - src.cols(), Core.BORDER_CONSTANT, Scalar.all(0));

        List<Mat> planes = new ArrayList<>();
        padded.convertTo(padded, CvType.CV_32F);
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        Mat complexImage = new Mat();
        Core.merge(planes, complexImage);

        Core.dft(complexImage, complexImage);

        Mat mask = createButterworthLowPassFilter(complexImage.size(), D0, n);

        // Apply the mask
        Core.split(complexImage, planes);
        Core.multiply(planes.get(0), mask, planes.get(0));
        Core.multiply(planes.get(1), mask, planes.get(1));
        Core.merge(planes, complexImage);

        // Perform inverse Fourier Transform
        Core.idft(complexImage, complexImage, Core.DFT_SCALE | Core.DFT_REAL_OUTPUT, 0);

        // Convert back to an image
        Mat result = new Mat();
        Core.split(complexImage, planes);
        Core.normalize(planes.get(0), result, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);

        // Set the filtered image to the ImageView
        imageView.setImage(convertMatToImage(result));
        return imageView;
    }

    private static Mat createButterworthLowPassFilter(Size size, double D0, int n) {
        Mat mask = Mat.zeros(size, CvType.CV_32F);
        Point center = new Point(size.width / 2, size.height / 2);

        for (int i = 0; i < size.height; i++) {
            for (int j = 0; j < size.width; j++) {
                double d = Math.sqrt(Math.pow(i - center.y, 2) + Math.pow(j - center.x, 2));
                double value = 1 / (1 + Math.pow(d / D0, 2 * n));
                mask.put(i, j, value);
            }
        }

        return mask;
    }

    public static ImageView applyHighButterworthFilter(ImageView imageView, double D0) {
        // Convert JavaFX Image to OpenCV Mat
        Mat src = convertImageToMat(imageView.getImage());

        // Convert to grayscale
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        // Perform Fourier Transform
        Mat padded = new Mat();
        int m = Core.getOptimalDFTSize(src.rows());
        int n = 3;
        Core.copyMakeBorder(src, padded, 0, m - src.rows(), 0, n - src.cols(), Core.BORDER_CONSTANT, Scalar.all(0));

        List<Mat> planes = new ArrayList<>();
        padded.convertTo(padded, CvType.CV_32F);
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        Mat complexImage = new Mat();
        Core.merge(planes, complexImage);

        Core.dft(complexImage, complexImage);

        // Create Butterworth high-pass filter mask
        Mat mask = createButterworthHighPassFilter(complexImage.size(), D0, n);

        // Apply the mask
        Core.split(complexImage, planes);
        Core.multiply(planes.get(0), mask, planes.get(0));
        Core.multiply(planes.get(1), mask, planes.get(1));
        Core.merge(planes, complexImage);

        // Perform inverse Fourier Transform
        Core.idft(complexImage, complexImage, Core.DFT_SCALE | Core.DFT_REAL_OUTPUT, 0);

        // Convert back to an image
        Mat result = new Mat();
        Core.split(complexImage, planes);
        Core.normalize(planes.get(0), result, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);

        // Set the filtered image to the ImageView
        imageView.setImage(convertMatToImage(result));
        return imageView;
    }

    private static Mat createButterworthHighPassFilter(Size size, double D0, int n) {
        Mat mask = Mat.ones(size, CvType.CV_32F);
        Point center = new Point(size.width / 2, size.height / 2);

        for (int i = 0; i < size.height; i++) {
            for (int j = 0; j < size.width; j++) {
                double d = Math.sqrt(Math.pow(i - center.y, 2) + Math.pow(j - center.x, 2));
                double value = 1 / (1 + Math.pow(D0 / d, 2 * n));
                mask.put(i, j, value);
            }
        }

        return mask;
    }
}