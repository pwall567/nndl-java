/*
 * @(#) MNISTImageData.java
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

package net.pwall.neural.test.images;

import java.io.IOException;

/**
 * MNIST Image Data.
 */
public class MNISTImageData extends MNISTData {

    private int numImages;
    private int numRows;
    private int numCols;
    private int numPixels;
    private byte[] array;

    public MNISTImageData(String filename) throws IOException {
        super(filename, 0x0803);
        numImages = read32();
        numRows = read32();
        numCols = read32();
        numPixels = numRows * numCols;
        array = new byte[numImages * numPixels];
        readArray(array, numImages * numPixels);
    }

    public int getNumImages() {
        return numImages;
    }

    public int getNumRows() {
        return numRows;
    }

    public int getNumCols() {
        return numCols;
    }

    public int getPixelValue(int image, int row, int col) {
        return array[image * numPixels + row * numCols + col] & 0xFF;
    }

    public int getPixelValue(int image, int pixel) {
        return array[image * numPixels + pixel] & 0xFF;
    }

}
