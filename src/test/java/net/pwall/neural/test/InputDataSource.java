/*
 * @(#) InputDataSource.java
 *
 * nndl-java Neural Networks and Deep Learning
 * Copyright (c) 2018 Peter Wall
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package net.pwall.neural.test;

import net.pwall.neural.TrainingData;
import net.pwall.neural.TrainingDataSource;
import net.pwall.neural.test.images.MNISTImageData;
import net.pwall.neural.test.images.MNISTLabelData;

/**
 * Input data class for neural network experimentation.
 *
 * @author  Peter Wall
 */
public class InputDataSource implements TrainingDataSource {

    private MNISTImageData imageData;
    private MNISTLabelData labelData;
    private int pixels;

    public InputDataSource(MNISTImageData imageData, MNISTLabelData labelData) {
        this.imageData = imageData;
        this.labelData = labelData;
        pixels = imageData.getNumRows() * imageData.getNumRows();
    }


    @Override
    public TrainingData getItem(int index) {
        return new InputData(index);
    }

    @Override
    public int getSize() {
        return imageData.getNumImages();
    }

    public class InputData implements TrainingData {

        private int image;

        public InputData(int image) {
            this.image = image;
        }

        @Override
        public double[] getInputs() {
            double[] result = new double[pixels];
            for (int i = 0, n = pixels; i < n; i++)
                result[i] = (double)imageData.getPixelValue(image, i) / 256;
            return result;
        }

        @Override
        public double[] getOutputs() {
            double[] result = new double[10];
            result[labelData.getLabelValue(image)] = 1.0;
            return result;
        }

    }

}
