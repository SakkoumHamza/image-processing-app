package com.example.imageprocessingapp;

import javafx.scene.image.*;
import javafx.fxml.*;
import javafx.scene.control.*;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Arrays;

public class Controller {

    @FXML private ImageView originalImageView;
    @FXML private ImageView processedImageView;
    @FXML private Button saveButton;

/*      ------------------------------------------- Opérations ------------------------------------------------        */

    @FXML
    private void onOpenButtonClick() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Choisir une image");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg", "*.bmp"));

        File selectedFile = fileChooser.showOpenDialog(originalImageView.getScene().getWindow());
        if (selectedFile != null) {
            try {
                // Load and display the image
                Image image = new Image(new FileInputStream(selectedFile));
                originalImageView.setImage(image);  // Show the image in the ImageView
                System.out.println("Image loaded: " + selectedFile.getAbsolutePath());
            } catch (FileNotFoundException e) {
                System.err.println("Error loading file: " + e.getMessage());
            }
        } else {
            System.out.println("No file selected!");
        }
    }
    @FXML
    private void onSaveButtonClick() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Enregistrer l'image traité ");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg", "*.bmp"));

        File saveFile = fileChooser.showSaveDialog(saveButton.getScene().getWindow());
        if (saveFile != null) {
            System.out.println("Saving processed image to: " + saveFile.getAbsolutePath());
            // Save the processed image
        } else {
            System.out.println("Save action canceled!");
        }
    }
    @FXML
    private void onQuitButtonClick() {
        Alert alert = new Alert(Alert.AlertType.CONFIRMATION, "Vous êtes sûr que vous voulez quitter !", ButtonType.YES, ButtonType.NO);
        alert.showAndWait();

        if (alert.getResult() == ButtonType.YES) {
            System.exit(0);
        }
    }

/*     ------------------------------------------- Filtres de fourier ------------------------------------------------        */

    @FXML
    private void onLowIdealButtonClick() {
        if (originalImageView.getImage() == null) {
            return;
        }
        ImageView lowIdealImage =FourierFilter.applyIdealLowPassFilter(originalImageView,30);
        processedImageView.setImage(lowIdealImage.getImage());
    }

    @FXML
    private void onHighIdealButtonClick() {
        if (originalImageView.getImage() == null) {
            return;
        }
        ImageView HighIdealImage =FourierFilter.applyIdealHighPassFilter(originalImageView,30);
        processedImageView.setImage(HighIdealImage.getImage());
    }
    @FXML
    private void onLowButterworthButtonClick() {
        if (originalImageView.getImage() == null) {
            return;
        }
        ImageView LowButterworthImage =FourierFilter.applyLowButterworthFilter(originalImageView,30);
        processedImageView.setImage(LowButterworthImage.getImage());
    }
    @FXML
    private void onHighButterworthButtonClick() {
        if (originalImageView.getImage() == null) {
            return;
        }
        ImageView HighButterworthImage =FourierFilter.applyHighButterworthFilter(originalImageView,30);
        processedImageView.setImage(HighButterworthImage.getImage());
    }

/*      ------------------------------------------- Filtres Pass-Haut ------------------------------------------------        */

    @FXML
        private void onLaplacienButtonClick() {
            if (originalImageView.getImage() == null) {
                return;
            }

            Image laplacianImage = applyLaplacianFilter(originalImageView.getImage());
            processedImageView.setImage(laplacianImage);
        }

        private Image applyLaplacianFilter(Image image) {
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();

            // Laplacian kernel
            int[][] laplacianKernel = {
                    {0, 1, 0},
                    {1, -4, 1},
                    {0, 1, 0}
            };

            // Apply Laplacian filter
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    double sum = 0;

                    // Convolution
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            Color color = pixelReader.getColor(x + kx, y + ky);
                            double gray = (color.getRed() + color.getGreen() + color.getBlue()) / 3.0;
                            sum += gray * laplacianKernel[ky + 1][kx + 1];
                        }
                    }

                    // Normalize the result
                    sum = Math.min(1, Math.max(0, sum));

                    // Set pixel color
                    Color newColor = new Color(sum, sum, sum, 1);
                    pixelWriter.setColor(x, y, newColor);
                }
            }

            return outputImage;
        }
    @FXML
        private void onSobelButtonClick() {
            // Check if an image is loaded
            if (originalImageView.getImage() == null) {
                return;
            }

            // Apply Sobel filter
            Image sobelImage = applySobelFilter(originalImageView.getImage());
            processedImageView.setImage(sobelImage);
        }

        private Image applySobelFilter(Image image) {
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();

            // Sobel kernels
            int[][] gx = {
                    {-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}
            };

            int[][] gy = {
                    {1, 2, 1},
                    {0, 0, 0},
                    {-1, -2, -1}
            };

            // Apply Sobel filter
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    double sumX = 0;
                    double sumY = 0;

                    // Convolution
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            Color color = pixelReader.getColor(x + kx, y + ky);
                            double gray = (color.getRed() + color.getGreen() + color.getBlue()) / 3.0;
                            sumX += gray * gx[ky + 1][kx + 1];
                            sumY += gray * gy[ky + 1][kx + 1];
                        }
                    }

                    // Calculate magnitude
                    double magnitude = Math.sqrt(sumX * sumX + sumY * sumY);
                    magnitude = Math.min(1, magnitude); // Normalize to [0, 1]

                    // Set pixel color
                    Color newColor = new Color(magnitude, magnitude, magnitude, 1);
                    pixelWriter.setColor(x, y, newColor);
                }
            }

            return outputImage;
        } 
        
    @FXML
        private void onGradientButtonClick() {
            if (originalImageView.getImage() == null) {
                return;
            }
        
            Image gradientImage = applyGradientFilter(originalImageView.getImage());
            processedImageView.setImage(gradientImage);
        }
        
        private Image applyGradientFilter(Image image) {
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();
        
            // Gradient kernels
            int[][] gx = {
                {-1, 0, 1},
                {-1, 0, 1},
                {-1, 0, 1}
            };
        
            int[][] gy = {
                {-1, -1, -1},
                {0, 0, 0},
                {1, 1, 1}
            };
        
            // Apply gradient filter
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    double sumX = 0;
                    double sumY = 0;
        
                    // Convolution
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            Color color = pixelReader.getColor(x + kx, y + ky);
                            double gray = (color.getRed() + color.getGreen() + color.getBlue()) / 3.0;
                            sumX += gray * gx[ky + 1][kx + 1];
                            sumY += gray * gy[ky + 1][kx + 1];
                        }
                    }
        
                    // Calculate magnitude
                    double magnitude = Math.sqrt(sumX * sumX + sumY * sumY);
                    magnitude = Math.min(1, magnitude); // Normalize to [0, 1]
        
                    // Set pixel color
                    Color newColor = new Color(magnitude, magnitude, magnitude, 1);
                    pixelWriter.setColor(x, y, newColor);
                }
            }
        
            return outputImage;
        }
    @FXML
        private void onPrewittButtonClick(){
            if (originalImageView.getImage() == null) {
                return;
            }
        
            Image prewittImage = applyPrewittFilter(originalImageView.getImage());
            processedImageView.setImage(prewittImage);
        }
        
        private Image applyPrewittFilter(Image image) {
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();
        
            // Prewitt kernels
            int[][] gx = {
                {-1, 0, 1},
                {-1, 0, 1},
                {-1, 0, 1}
            };
        
            int[][] gy = {
                {-1, -1, -1},
                {0, 0, 0},
                {1, 1, 1}
            };
        
            // Apply Prewitt filter
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    double sumX = 0;
                    double sumY = 0;
        
                    // Convolution
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            Color color = pixelReader.getColor(x + kx, y + ky);
                            double gray = (color.getRed() + color.getGreen() + color.getBlue()) / 3.0;
                            sumX += gray * gx[ky + 1][kx + 1];
                            sumY += gray * gy[ky + 1][kx + 1];
                        }
                    }
        
                    // Calculate magnitude
                    double magnitude = Math.sqrt(sumX * sumX + sumY * sumY);
                    magnitude = Math.min(1, magnitude); // Normalize to [0, 1]
        
                    // Set pixel color
                    Color newColor = new Color(magnitude, magnitude, magnitude, 1);
                    pixelWriter.setColor(x, y, newColor);
                }
            }
            return outputImage;
        }
    @FXML
        private void onRobertsButtonClick(){
            if (originalImageView.getImage() == null) {
                return;
            }
        
            Image robertsImage = applyRobertsFilter(originalImageView.getImage());
            processedImageView.setImage(robertsImage);
        }
        
        private Image applyRobertsFilter(Image image) {
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();
        
            // Roberts kernels
            int[][] gx = {
                {1, 0},
                {0, -1}
            };
        
            int[][] gy = {
                {0, 1},
                {-1, 0}
            };
        
            // Apply Roberts filter
            for (int y = 0; y < height - 1; y++) {
                for (int x = 0; x < width - 1; x++) {
                    double sumX = 0;
                    double sumY = 0;
        
                    // Convolution
                    for (int ky = 0; ky <= 1; ky++) {
                        for (int kx = 0; kx <= 1; kx++) {
                            Color color = pixelReader.getColor(x + kx, y + ky);
                            double gray = (color.getRed() + color.getGreen() + color.getBlue()) / 3.0;
                            sumX += gray * gx[ky][kx];
                            sumY += gray * gy[ky][kx];
                        }
                    }
        
                    // Calculate magnitude
                    double magnitude = Math.sqrt(sumX * sumX + sumY * sumY);
                    magnitude = Math.min(1, magnitude); // Normalize to [0, 1]
        
                    // Set pixel color
                    Color newColor = new Color(magnitude, magnitude, magnitude, 1);
                    pixelWriter.setColor(x, y, newColor);
                }
            }
            return outputImage;
        }


/*      ------------------------------------------- Filtres Pass-Bas ------------------------------------------------        */

    @FXML
        private void onMoyenneur33ButtonClick(){
            if (originalImageView.getImage() == null) {
                return;
            }
        
            Image meanImage = applyMeanFilter(originalImageView.getImage(), 3);
            processedImageView.setImage(meanImage);
        }
        
        private Image applyMeanFilter(Image image, int kernelSize) {
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();
        
            int offset = kernelSize / 2;
        
            // Apply mean filter
            for (int y = offset; y < height - offset; y++) {
                for (int x = offset; x < width - offset; x++) {
                    double sumRed = 0;
                    double sumGreen = 0;
                    double sumBlue = 0;
        
                    // Convolution
                    for (int ky = -offset; ky <= offset; ky++) {
                        for (int kx = -offset; kx <= offset; kx++) {
                            Color color = pixelReader.getColor(x + kx, y + ky);
                            sumRed += color.getRed();
                            sumGreen += color.getGreen();
                            sumBlue += color.getBlue();
                        }
                    }
        
                    // Calculate average
                    int kernelArea = kernelSize * kernelSize;
                    double avgRed = sumRed / kernelArea;
                    double avgGreen = sumGreen / kernelArea;
                    double avgBlue = sumBlue / kernelArea;
        
                    // Set pixel color
                    Color newColor = new Color(avgRed, avgGreen, avgBlue, 1);
                    pixelWriter.setColor(x, y, newColor);
                }
            }
        
            return outputImage;
        }
    @FXML
        private void onMoyenneur55ButtonClick(){
            if (originalImageView.getImage() == null) {
                return;
            }
        
            Image meanImage = applyMeanFilter(originalImageView.getImage(), 5);
            processedImageView.setImage(meanImage);
        }
    @FXML
        private void onGaussien33ButtonClick(){
            if (originalImageView.getImage() == null) {
                return;
            }
        
            Image gaussianImage = applyGaussianFilter(originalImageView.getImage(), 3);
            processedImageView.setImage(gaussianImage);
        }
        
        private Image applyGaussianFilter(Image image, int kernelSize) {
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();
        
            int offset = kernelSize / 2;
        
            // Gaussian kernel (3x3)
            double[][] gaussianKernel = {
                {1, 2, 1},
                {2, 4, 2},
                {1, 2, 1}
            };
        
            // Normalize kernel
            double kernelSum = 16; // Sum of the kernel values
        
            // Apply Gaussian filter
            for (int y = offset; y < height - offset; y++) {
                for (int x = offset; x < width - offset; x++) {
                    double sumRed = 0;
                    double sumGreen = 0;
                    double sumBlue = 0;
        
                    // Convolution
                    for (int ky = -offset; ky <= offset; ky++) {
                        for (int kx = -offset; kx <= offset; kx++) {
                            Color color = pixelReader.getColor(x + kx, y + ky);
                            double weight = gaussianKernel[ky + offset][kx + offset];
                            sumRed += color.getRed() * weight;
                            sumGreen += color.getGreen() * weight;
                            sumBlue += color.getBlue() * weight;
                        }
                    }
        
                    // Normalize
                    double avgRed = sumRed / kernelSum;
                    double avgGreen = sumGreen / kernelSum;
                    double avgBlue = sumBlue / kernelSum;
        
                    // Set pixel color
                    Color newColor = new Color(avgRed, avgGreen, avgBlue, 1);
                    pixelWriter.setColor(x, y, newColor);
                }
            }
        
            return outputImage;
        }
    @FXML
        private void onGaussien55ButtonClick(){
            if (originalImageView.getImage() == null) {
                return;
            }
        
            Image gaussianImage = applyGaussianFilter(originalImageView.getImage(), 5);
            processedImageView.setImage(gaussianImage);
        }
    @FXML
        private void onMedianeButtonClick(){
            if (originalImageView.getImage() == null) {
                    return;
                }
            Image medianImage = applyMedianFilter(originalImageView.getImage(), 3);
            processedImageView.setImage(medianImage);
        }

        public static Image applyMedianFilter(Image image, int kernelSize) {
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();
        
            int offset = kernelSize / 2;
            int windowSize = kernelSize * kernelSize;
            
            // Apply median filter
            for (int y = offset; y < height - offset; y++) {
                for (int x = offset; x < width - offset; x++) {
                    double[] redValues = new double[windowSize];
                    double[] greenValues = new double[windowSize];
                    double[] blueValues = new double[windowSize];
                    int index = 0;
        
                    // Collect values from the neighborhood
                    for (int ky = -offset; ky <= offset; ky++) {
                        for (int kx = -offset; kx <= offset; kx++) {
                            Color color = pixelReader.getColor(x + kx, y + ky);
                            redValues[index] = color.getRed();
                            greenValues[index] = color.getGreen();
                            blueValues[index] = color.getBlue();
                            index++;
                        }
                    }
        
                    // Sort values
                    Arrays.sort(redValues);
                    Arrays.sort(greenValues);
                    Arrays.sort(blueValues);
        
                    // Get median values (middle of sorted arrays)
                    int medianIndex = windowSize / 2;
                    Color medianColor = Color.color(
                        redValues[medianIndex],
                        greenValues[medianIndex],
                        blueValues[medianIndex]
                    );
        
                    pixelWriter.setColor(x, y, medianColor);
                }
            }
        
            // Handle borders by copying edge pixels
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (y < offset || y >= height - offset || x < offset || x >= width - offset) {
                        Color color = pixelReader.getColor(
                            Math.min(Math.max(x, offset), width - offset - 1),
                            Math.min(Math.max(y, offset), height - offset - 1)
                        );
                        pixelWriter.setColor(x, y, color);
                    }
                }
            }
        
            return outputImage;
        }

/*      ------------------------------------------- Transformations ------------------------------------------------        */

    @FXML
        private void onBinarisationButtonClick() {
            if (originalImageView.getImage() == null) {
                return;
            }
        
            Image image = originalImageView.getImage();
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();
            
            // Threshold value (can be adjusted)
            double threshold = 0.5;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Color color = pixelReader.getColor(x, y);
                    // Convert to grayscale first
                    double gray = (color.getRed() + color.getGreen() + color.getBlue()) / 3.0;
                    // Apply threshold
                    Color newColor = gray > threshold ? Color.WHITE : Color.BLACK;
                    pixelWriter.setColor(x, y, newColor);
                }
            }
            
            processedImageView.setImage(outputImage);
        }
        
    @FXML
        private void onContrasteButtonClick() {
            if (originalImageView.getImage() == null) {
                System.out.println("No image loaded");
                return;
            }
            
            Image image = originalImageView.getImage();
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();
            
            // Contrast factor (adjust as needed)
            double factor = 1.5;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Color color = pixelReader.getColor(x, y);
                    
                    // Apply contrast adjustment
                    double red = adjustContrast(color.getRed(), factor);
                    double green = adjustContrast(color.getGreen(), factor);
                    double blue = adjustContrast(color.getBlue(), factor);
                    
                    Color newColor = Color.color(red, green, blue);
                    pixelWriter.setColor(x, y, newColor);
                }
            }
            
            processedImageView.setImage(outputImage);
        }
        
        private double adjustContrast(double value, double factor) {
            double adjusted = (value - 0.5) * factor + 0.5;
            return Math.min(1.0, Math.max(0.0, adjusted));
        }
        
    @FXML
        private void onDivisionButtonClick() {
            if (originalImageView.getImage() == null) {
                System.out.println("No image loaded");
                return;
            }
            
            Image image = originalImageView.getImage();
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();
            
            // Division factor (adjust as needed)
            double divisionFactor = 2.0;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Color color = pixelReader.getColor(x, y);
                    
                    Color newColor = Color.color(
                        color.getRed() / divisionFactor,
                        color.getGreen() / divisionFactor,
                        color.getBlue() / divisionFactor
                    );
                    
                    pixelWriter.setColor(x, y, newColor);
                }
            }
            
            processedImageView.setImage(outputImage);
        }
        
    @FXML
        private void onNvGrayButtonClick() {
            if (originalImageView.getImage() == null) {
                System.out.println("No image loaded");
                return;
            }
            
            Image image = originalImageView.getImage();
            int width = (int) image.getWidth();
            int height = (int) image.getHeight();
            WritableImage outputImage = new WritableImage(width, height);
            PixelWriter pixelWriter = outputImage.getPixelWriter();
            PixelReader pixelReader = image.getPixelReader();
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Color color = pixelReader.getColor(x, y);
                    
                    // Convert to grayscale using luminance formula
                    double gray = 0.299 * color.getRed() + 
                                 0.587 * color.getGreen() + 
                                 0.114 * color.getBlue();
                    
                    Color grayColor = Color.color(gray, gray, gray);
                    pixelWriter.setColor(x, y, grayColor);
                }
            }
            
            processedImageView.setImage(outputImage);
        }

    @FXML
        private void onHistogramButtonClick(){
            // Check if an image is loaded
            if (originalImageView.getImage() == null) {
                return;
            }

            // Get image pixels
            PixelReader pixelReader = originalImageView.getImage().getPixelReader();
            int width = (int) originalImageView.getImage().getWidth();
            int height = (int) originalImageView.getImage().getHeight();

            // Initialize histograms for RGB channels
            int[] redHistogram = new int[256];
            int[] greenHistogram = new int[256];
            int[] blueHistogram = new int[256];

            // Calculate histogram values
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Color color = pixelReader.getColor(x, y);
                    redHistogram[(int) (color.getRed() * 255)]++;
                    greenHistogram[(int) (color.getGreen() * 255)]++;
                    blueHistogram[(int) (color.getBlue() * 255)]++;
                }
            }

            // Create histogram image
            int histHeight = 200;
            WritableImage histogramImage = new WritableImage(256, histHeight);
            PixelWriter pixelWriter = histogramImage.getPixelWriter();

            // Find maximum frequency for scaling
            int maxFreq = 0;
            for (int i = 0; i < 256; i++) {
                maxFreq = Math.max(maxFreq, Math.max(redHistogram[i],
                        Math.max(greenHistogram[i], blueHistogram[i])));
            }

            // Draw histogram
            for (int x = 0; x < 256; x++) {
                int redHeight = (int) ((redHistogram[x] * histHeight) / (double) maxFreq);
                int greenHeight = (int) ((greenHistogram[x] * histHeight) / (double) maxFreq);
                int blueHeight = (int) ((blueHistogram[x] * histHeight) / (double) maxFreq);

                for (int y = 0; y < histHeight; y++) {
                    if (y >= histHeight - redHeight) {
                        pixelWriter.setColor(x, y, Color.RED.deriveColor(1, 1, 1, 0.3));
                    }
                    if (y >= histHeight - greenHeight) {
                        pixelWriter.setColor(x, y, Color.GREEN.deriveColor(1, 1, 1, 0.3));
                    }
                    if (y >= histHeight - blueHeight) {
                        pixelWriter.setColor(x, y, Color.BLUE.deriveColor(1, 1, 1, 0.3));
                    }
                }
            }

            // Display histogram in processedImageView
            processedImageView.setImage(histogramImage);
        }


}