/*
 * @(#) TrainingData.java
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
 * Training data for Stochastic Gradient Descent.  Implementations of this interface must supply
 * two methods: {@link #getInputs()} to get the inputs in the form of an array of {@code double}
 * in the range 0.0 to 1.0, and {@link #getOutputs()} to get the corresponding set of expected
 * outputs.  Where the index of the highest output is known (for example, when the output array
 * has a single value set to 1.0) an implementation of {@link #getHighestOutputIndex()} should
 * be provided as an optimisation.
 *
 * @author Peter Wall
 */
public interface TrainingData {

    /**
     * Get a set of inputs for training.
     *
     * @return  the set of inputs as a {@code double} array
     */
    double[] getInputs();

    /**
     * Get the expected outputs for training.
     *
     * @return  the set of expected outputs as a {@code double} array
     */
    double[] getOutputs();

    /**
     * Get the expected output as an integer index (the index of the highest value in the output
     * array).
     *
     * @return  the expected output integer
     */
    default int getHighestOutputIndex() {
        return Network.indexOfHighest(getOutputs());
    }

}
