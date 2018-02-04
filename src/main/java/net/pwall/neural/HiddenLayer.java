/*
 * @(#) HiddenLayer.java
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

import java.util.Objects;
import java.util.Random;



/**
 * Class to represent a hidden layer of a neural network.
 *
 * @author Peter Wall
 */
public class HiddenLayer implements Layer {

    private Layer input;
    private int size;
    private int inputSize;
    private double[][] weights;
    private double[] biases;
    private double[] outputs;

    /**
     * Construct the input layer with the required size.
     *
     * @param   input   the input layer (may be another hidden layer)
     * @param   size    the number of inputs
     */
    public HiddenLayer(Layer input, int size) {
        this.input = Objects.requireNonNull(input);
        this.size = checkSize(size);
        inputSize = input.getSize();
        weights = getZeroWeightsArray();
        biases = getZeroBiasesArray();
        outputs = new double[size];
    }

    /**
     * Initialise the layer using the supplied {@link Random}.  The {@code Random} is an
     * argument to allow the user to supply a {@code Random} with a known seed for repeatable
     * results.
     *
     * @param   r       the {@link Random}
     */
    public void init(Random r) {
        for (int i = 0; i < weights.length; i++) {
            double[] weightsi = weights[i];
            for (int j = 0; j < inputSize; j++)
                weightsi[j] = r.nextGaussian();
            biases[i] = r.nextGaussian();
        }
    }

    /**
     * Perform a single iteration of the layer.  In theory this operation could be parallelised,
     * but in practice the performance cost of setting up the parallel stream would greatly
     * outweigh the possible benefits.
     */
    public void iterate() {
        double[] inputs = input.getOutputs();

        // iterate over each neuron

        for (int i = 0; i < weights.length; i++) {
            double[] neuronWeights = weights[i];
            double sum = 0.0;

            // iterate over inputs to neuron

            for (int j = 0; j < inputSize; j++)
                sum += inputs[j] * neuronWeights[j];

            outputs[i] = activation(sum + biases[i]);
        }
    }

    /**
     * Get the input layer to this layer.  That may be an {@link InputLayer} or another
     *  {@link HiddenLayer}
     *
     * @return  the input layer
     */
    public Layer getInput() {
        return input;
    }

    /**
     * Get the biases as an array.  For performance reasons this method returns a reference to
     * the original array rather than a copy; the array must be treated as immutable.
     *
     * @return  the biases
     */
    public double[] getBiases() {
        return biases;
    }

    /**
     * Set the biases from a supplied array.
     *
     * @param   newBiases       the new biases
     * @throws  IllegalArgumentException if the array is of the wrong size
     */
    public void setBiases(double[] newBiases) {
        if (newBiases.length != size)
            throw new IllegalArgumentException("Wrong size");
        System.arraycopy(newBiases, 0, biases, 0, size);
    }

    /**
     * Get the weights as a 2-dimension array.  For performance reasons this method returns a
     * reference to the original array rather than a copy; the array must be treated as
     * immutable.
     *
     * @return  the weights
     */
    public double[][] getWeights() {
        return weights;
    }

    /**
     * Set the weights from a supplied 2-dimension array.
     *
     * @param   newWeights      the new weights
     * @throws  IllegalArgumentException if the array is of the wrong dimensions
     */
    public void setWeights(double[][] newWeights) {
        if (newWeights.length != size || newWeights[0].length != inputSize)
            throw new IllegalArgumentException("Wrong size");
        for (int i = 0; i < size; i++)
            System.arraycopy(newWeights[i], 0, weights[i], 0, inputSize);
    }

    /**
     * Get a zero array of the same size as the biases array.
     *
     * @return  the zero array
     */
    public double[] getZeroBiasesArray() {
        return new double[size];
    }

    /**
     * Get a zero array of the same dimensions as the weights array.
     *
     * @return  the zero array
     */
    public double[][] getZeroWeightsArray() {
        double[][] result = new double[size][];
        for (int i = 0; i < size; i++)
            result[i] = new double[inputSize];
        return result;
    }

    /**
     * The activation function.  This is a separate method so that it may be overridden in a
     * subclass.
     *
     * @param   d       the raw value
     * @return  the activation function value
     */
    public double activation(double d) {
        return 1.0 / (1.0 + Math.exp(-d)); // sigmoid function
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
     * Get the outputs as an array.  For performance reasons this method returns a reference to
     * the original array rather than a copy; the array must be treated as immutable.
     *
     * @return  the outputs
     */
    @Override
    public double[] getOutputs() {
        return outputs;
    }

}
