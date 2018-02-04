/*
 * @(#) InputLayer.java
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

package net.pwall.neural;



/**
 * Class to represent the input layer of a neural network.
 *
 * @author Peter Wall
 */
public class InputLayer implements Layer {

    private int size;
    private double[] values;

    /**
     * Construct the input layer with the required size.
     *
     * @param   size    the number of inputs
     */
    public InputLayer(int size) {
        this.size = checkSize(size);
        values = new double[size];
    }

    /**
     * Get the size of the layer.
     *
     * @return  the size
     */
    @Override
    public int getSize() {
        return size;
    }

    /**
     * Get the output values as an array.
     */
    @Override
    public double[] getOutputs() {
        return values;
    }

    /**
     * Set the value of an individual input.
     *
     * @param   index   the index of the value
     * @param   value   the new value
     */
    public void setValue(int index, double value) {
        values[index] = value;
    }

    /**
     * Set the values of all the inputs in a single operation.
     *
     * @param   values   the new values
     */
    public void setValues(double[] values) {
        if (values == null || values.length != size)
            throw new IllegalArgumentException("Input layer values array null or wrong size");
        System.arraycopy(values, 0, this.values, 0, size);
    }

}
