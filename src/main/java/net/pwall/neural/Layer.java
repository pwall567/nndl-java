/*
 * @(#) Layer.java
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
 * Interface for layers in a neural network.
 *
 * @author Peter Wall
 */
public interface Layer {

    /**
     * Get the size of the layer.
     *
     * @return  the layer size
     */
    int getSize();

    /**
     * Get the output values as an array of {@code double}.
     *
     * @return  the output values
     */
    double[] getOutputs();

    /**
     * Get an individual output value.  This default implementation assumes that
     * {@link #getOutputs()} returns a reference to the actual array, rather than a copy.  If
     * this is not the case, this method will be very slow and an alternative implementation
     * should be provided.
     *
     * @param   index   the index of the value
     * @return  the output value
     */
    default double getOutput(int index) {
        return getOutputs()[index];
    }

    /**
     * Check layer size is positive.
     *
     * @param   size    the layer size
     * @return  the size
     * @throws  IllegalArgumentException if the size is &lt;= 0
     */
    default int checkSize(int size) {
        if (size <= 0)
            throw new IllegalArgumentException("Layer size must be >= 0");
        return size;
    }

}
