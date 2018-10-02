/*
 * @(#) Network.java
 *
 * nndl-java Neural Networks and Deep Learning
 * Copyright (c) 2018 Peter Wall
 * Derived from original Python code copyright (c) 2012-2015 Michael Nielsen
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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;



/**
 * Neural Network.  This implementation is based on the work of Michael Nielsen in the online
 * book <a href="http://neuralnetworksanddeeplearning.com/">Neural Networks and Deep
 * Learning</a>.
 *
 * <p>I cannot speak highly enough of this book, which gives a clear explanation of the theory
 * and mathematics of neural networks as well as implementation code in Python which formed the
 * basis of this Java version.</p>
 *
 * @author      Peter Wall
 * @author      Michael Nielsen (original Python code)
 */
public class Network {

    private static final Log log = LogFactory.getLog(Network.class);

    private int numLayers;
    private InputLayer inputLayer;
    private HiddenLayer[] hiddenLayers;

    /**
     * Construct a network with the supplied layer sizes.
     *
     * @param   layerSizes      the layer sizes
     * @throws  IllegalArgumentException if the number of layers less than 2
     */
    public Network(int ... layerSizes) {
        numLayers = layerSizes.length;
        if (numLayers < 2)
            throw new IllegalArgumentException("Must have 2 or more layers");
        inputLayer = new InputLayer(layerSizes[0]);
        hiddenLayers = new HiddenLayer[numLayers - 1];
        hiddenLayers[0] = new HiddenLayer(inputLayer, layerSizes[1]);
        for (int i = 2; i < numLayers; i++)
            hiddenLayers[i - 1] = new HiddenLayer(hiddenLayers[i - 2], layerSizes[i]);
    }

    /**
     * Get the layer by number.
     *
     * @param   index   the layer number
     * @return  the layer
     */
    public Layer getLayer(int index) {
        if (index >= numLayers)
            throw new IndexOutOfBoundsException(String.valueOf(index));
        return index == 0 ? inputLayer : hiddenLayers[index - 1];
    }

    /**
     * Get the input layer.
     *
     * @return  the input layer
     */
    public InputLayer getInputLayer() {
        return inputLayer;
    }

    /**
     * Set the value of an individual input.
     *
     * @param   index   the index of the value
     * @param   value   the new value
     */
    public void setInput(int index, double value) {
        inputLayer.setValue(index, value);
    }

    /**
     * Set the values of all the inputs in a single operation.
     *
     * @param   values   the new values
     */
    public void setInputs(double[] values) {
        inputLayer.setValues(values);
    }

    /**
     * Get the output layer (in this implementation all layers other than the input layer are
     * called hidden layers; the output layer is the last hidden layer).
     *
     * @return  the output layer
     */
    public HiddenLayer getOutputLayer() {
        return hiddenLayers[numLayers - 2];
    }

    /**
     * get the outputs of the output layer.
     *
     * @return  the outputs
     */
    public double[] getOutputs() {
        return getOutputLayer().getOutputs();
    }

    /**
     * Initialise the network using the supplied {@link Random}.  The {@code Random} is supplied
     * as an argument to allow the user to use a {@code Random} with a known seed for repeatable
     * results.
     *
     * @param   r   the {@link Random}
     */
    public void init(Random r) {
        for (int i = 0; i < numLayers - 1; i++)
            hiddenLayers[i].init(r);
    }

    /**
     * Initialise the network.
     */
    public void init() {
        init(new Random());
    }

    /**
     * Process an array of inputs to produce an array of outputs.  This is the principal
     * function of a neural network, but for most purposes the full array of outputs is not
     * required, just the index of the highest output.  For this usage, see
     * {@link #getResultInt(double[])}.
     *
     * @param   inputs  the array of inputs (no length checking is performed)
     * @return
     */
    public double[] getResultArray(double[] inputs) {
        setInputs(inputs);
        int numHiddens = hiddenLayers.length;
        // important - this can not be parallelised
        for (int i = 0; i < numHiddens; i++)
            hiddenLayers[i].iterate();
        return getOutputs();
    }

    /**
     * Process an array of inputs to get a single integer output - the index of the highest
     * value in the output array.
     *
     * @param   inputs  the array of inputs (no length checking is performed)
     * @return  the index of the highest output
     */
    public int getResultInt(double[] inputs) {
        return indexOfHighest(getResultArray(inputs));
    }

    /**
     * Implementation of the mini-batch Stochastic Gradient Descent algorithm.
     *
     * See <a href="http://neuralnetworksanddeeplearning.com/chap1.html">Neural Networks and
     * Deep Learning, Chapter 1</a> for a full description of this functionality.
     *
     * @param   tds             a {@link TrainingDataSource}
     * @param   epochs          number of epochs
     * @param   miniBatchSize   the size of a mini-batch
     * @param   eta             the learning rate
     * @param   r               a {@link Random}, used to shuffle the training data (see the
     *                          note on {@link #init(Random)}
     * @param   testData        a second {@link TrainingDataSource} containing test data to
     *                          evaluate progress (may be {@code null})
     * @throws  IllegalArgumentException if the number of epochs not in allowed range
     */
    public void stochasticGradientDescent(TrainingDataSource tds, int epochs, int miniBatchSize,
            double eta, Random r, TrainingDataSource testData) {
        if (log.isInfoEnabled()) {
            log.info("Stochastic Gradient Descent on " + toString() + "; training data " +
                    tds.getSize() + "; " + epochs + " epochs; mini-batch size " +
                    miniBatchSize + "; eta " + eta);
        }
        if (r == null)
            r = new Random();
        TrainingDataRandom tdr = new TrainingDataRandom(Objects.requireNonNull(tds));
        if (epochs < 1 || epochs > 200)
            throw new IllegalArgumentException("number of epochs must be in range 1..200");
        long startTime = System.currentTimeMillis();
        for (int epoch = 0; epoch < epochs; epoch++) {
            tdr.randomise(r);
            for (int k = 0; k < tdr.getSize(); k += miniBatchSize) {
                TrainingDataSubset miniBatch = new TrainingDataSubset(tdr, k,
                        Math.min(miniBatchSize, tdr.getSize() - k));
                updateMiniBatch(miniBatch, eta);
            }
            if (log.isInfoEnabled()) {
                long now = System.currentTimeMillis();
                log.info("Completed epoch " + (epoch + 1) + " (" + (now - startTime) + "ms)");
                startTime = now;
                if (testData != null) {
                    int n = evaluate(testData);
                    now = System.currentTimeMillis();
                    log.info("Correctly identified " + n + " of " + testData.getSize() +
                            " (" + (now - startTime) + "ms)");
                    startTime = now;
                }
            }
        }
    }

    /**
     * Implementation of "update_mini_batch".
     *
     * @param   miniBatch       the mini-batch
     * @param   eta             the learning rate
     */
    private void updateMiniBatch(TrainingDataSubset miniBatch, double eta) {
        int numHiddens = hiddenLayers.length;
        double[][] nablaB = new double[numHiddens][];
        double[][][] nablaW = new double[numHiddens][][];
        for (int i = 0; i < numHiddens; i++) {
            nablaB[i] = hiddenLayers[i].getZeroBiasesArray();
            nablaW[i] = hiddenLayers[i].getZeroWeightsArray();
        }

        double[][] deltaNablaB = new double[numHiddens][];
        double[][][] deltaNablaW = new double[numHiddens][][];
        for (int m = 0, n = miniBatch.getSize(); m < n; m++) {
            backProp(miniBatch.getItem(m), deltaNablaB, deltaNablaW);
            for (int i = 0; i < numHiddens; i++) {
                addInPlace(nablaB[i], deltaNablaB[i]);
                addInPlace2(nablaW[i], deltaNablaW[i]);
            }
        }

        double etaDivBatchSize = eta / miniBatch.getSize();
        for (int i = 0; i < numHiddens; i++) {
            HiddenLayer h = hiddenLayers[i];
            double[] biases = h.getBiases(); // reference, not copy
            double[] nablaBi = nablaB[i];
            double[][] weights = h.getWeights(); // reference, not copy
            double[][] nablaWi = nablaW[i];
            for (int j = 0; j < h.getSize(); j++) {
                // calculate new biases
                biases[j] -= etaDivBatchSize * nablaBi[j];
                // calculate new weights
                double[] weightsj = weights[j];
                double[] nablaWij = nablaWi[j];
                for (int k = 0; k < weightsj.length; k++)
                    weightsj[k] -= etaDivBatchSize * nablaWij[k];
            }
        }
    }

    /**
     * Calculate gradient for the cost function.  {@code nablaB} and {@code nablaW} are
     * layer-by-layer lists of {@code double[]} or {@code double[][]} arrays, similar to
     * {@code biases} and {@code weights}.
     *
     * @param   td      the training data item
     */
    private void backProp(TrainingData td, double[][] nablaB, double[][][] nablaW) {
        int numHiddens = hiddenLayers.length;

        // feedforward
        double[] activation = td.getInputs();
        double[][] activations = new double[numHiddens + 1][];
        activations[0] = activation;
        double[][] zs = new double[numHiddens][];
        for (int i = 0; i < numHiddens; i++) {
            double[] z = dot(hiddenLayers[i].getWeights(), activation);
            addInPlace(z, hiddenLayers[i].getBiases());
            zs[i] = z;
            activation = sigmoid(z);
            activations[i + 1] = activation;
        }

        // backward pass
        double[] delta = costDerivative(activations[numHiddens], td.getOutputs());
        multiplyInPlace(delta, sigmoidPrime(zs[zs.length - 1]));
        nablaB[numHiddens - 1] = delta;
        nablaW[numHiddens - 1] = matrixMultiply(delta, activations[numHiddens - 1]);

        for (int l = 2; l < numLayers; l++) {
            double[] z = zs[zs.length - l];
            double[] sp = sigmoidPrime(z);
            delta = dot(transpose(hiddenLayers[numHiddens - l + 1].getWeights()), delta);
            multiplyInPlace(delta, sp);
            nablaB[numHiddens - l] = delta;
            nablaW[numHiddens - l] = matrixMultiply(delta, activations[numHiddens - l]);
        }

    }

    public double[] costDerivative(double[] outputActivations, double[] y) {
        int n = outputActivations.length;
        if (n != y.length)
            throw arraySameLengthException(n, y.length);
        double[] result = new double[n];
        for (int i = 0; i < n; i++)
            result[i] = outputActivations[i] - y[i];
        return result;
    }

    /**
     * Return the number of test inputs for which the neural network outputs the correct result.
     * Note that the neural network's output is assumed to be the index of whichever neuron in
     * the final layer has the highest activation.
     *
     * @param   testData    the set of test data
     * @return              the total number of correct results
     */
    public int evaluate(TrainingDataSource testData) {
        // parallelise?
        int sum = 0;
        for (int i = 0, n = testData.getSize(); i < n; i++) {
            TrainingData td = testData.getItem(i);
            if (getResultInt(td.getInputs()) == td.getHighestOutputIndex())
                sum++;
        }
        return sum;
    }

    /**
     * Static method to find the index of the highest value in an array (useful for converting
     * an output array to a single integer value).  Note that for performance reasons no
     * checking is performed on the validity of the array argument.
     *
     * @param   a       an array of {@code double}
     * @return  the index of the highest value in the array
     */
    public static int indexOfHighest(double[] a) {
        int result = 0;
        double highest = a[0];
        for (int i = 1, n = a.length; i < n; i++) {
            if (a[i] > highest) {
                highest = a[i];
                result = i;
            }
        }
        return result;
    }

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double[] sigmoid(double[] a) {
        int n = a.length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++)
            result[i] = sigmoid(a[i]);
        return result;
    }

    public static double sigmoidPrime(double z) {
        double s = sigmoid(z);
        return s * (1.0 - s);
    }

    public static double[] sigmoidPrime(double[] a) {
        int n = a.length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++)
            result[i] = sigmoidPrime(a[i]);
        return result;
    }

    /**
     * Matrix multiplication of two 2-dimension arrays.  Do a web search for "matrix
     * multiplication" for an explanation.  The second dimension of the first array must equal
     * the first dimension of the second array.
     *
     * @param   a       the first array
     * @param   b       the second array
     * @return          the matrix product
     * @throws  IllegalArgumentException if the arrays are of incompatible dimensions
     */
    public static double[][] dot(double[][] a, double[][] b) {
        int n = a[0].length;
        if (n != b.length)
            throw new IllegalArgumentException("Arrays must be compatible (" + n + " != " +
                    b.length + ')');
        double[][] result = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            int m = b[0].length;
            result[i] = new double[m];
            for (int j = 0; j < m; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++)
                    sum += a[i][k] * b[k][j];
                result[i][j] = sum;
            }
        }
        return result;
    }

    /**
     * Matrix multiplication of a 2-dimension array by a 1-dimension array.  This is an
     * adaptation of {@link #dot(double[][], double[][])}.  It treats the second argument as if
     * it were a 2-dimension array where the second dimension is always 1.  The resulting array
     * would have a second dimension of 1, so this method returns an array of a single
     * dimension.
     *
     * @param   a       a 2-dimension array
     * @param   b       a 1-dimension array
     * @return          a 1-dimension array
     * @throws  IllegalArgumentException if the arrays are of incompatible dimensions
     */
    public static double[] dot(double[][] a, double[] b) {
        int n = a[0].length;
        if (n != b.length)
            throw arraySameLengthException(n, b.length);
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double sum = 0.0;
            double[] ai = a[i];
            for (int k = 0; k < n; k++)
                sum += ai[k] * b[k];
            result[i] = sum;
        }
        return result;
    }

    /**
     * This is a special case of a matrix multiply, where the first operand is treated as if it
     * were of dimension (x, 1) and the second (1, y).  The resulting array is of dimension
     * (x, y).
     *
     * @param   a       the first array
     * @param   b       the second array
     * @return          the matrix product
     */
    public static double[][] matrixMultiply(double[] a, double[] b) {
        int alen = a.length;
        int blen = b.length;
        double[][] result = new double[a.length][];
        for (int i = 0; i < alen; i++) {
            double[] newArray = new double[blen];
            result[i] = newArray;
            double aval = a[i];
            for (int j = 0; j < blen; j++)
                newArray[j] = aval * b[j];
        }
        return result;
    }

    /**
     * Transpose the dimensions of a 2-dimension array of {@code double}.
     *
     * @param   a       the array
     * @return  the transposed array
     */
    public static double[][] transpose(double[][] a) {
        int len1 = a.length;
        int len2 = a[0].length;
        double[][] result = new double[len2][];
        for (int i = 0; i < len2; i++) {
            double[] newArray = new double[len1];
            result[i] = newArray;
            for (int j = 0; j < len1; j++)
                newArray[j] = a[j][i];
        }
        return result;
    }

    /**
     * Add two arrays of {@code double}, overwriting the first.  The arrays must be the same
     * size.
     *
     * @param   a       the first array.
     * @param   b       the second array
     * @throws  IllegalArgumentException if the arrays are of different lengths
     */
    public static void addInPlace(double[] a, double[] b) {
        int n = a.length;
        if (n != b.length)
            throw arraySameLengthException(n, b.length);
        for (int i = 0; i < n; i++)
            a[i] += b[i];
    }

    /**
     * Add two 2-dimension arrays of {@code double}, overwriting the first.  The arrays must be
     * of the same dimensions.
     *
     * @param   a       the first array.
     * @param   b       the second array
     * @throws  IllegalArgumentException if the arrays are of different dimensions
     */
    public static void addInPlace2(double[][] a, double[][] b) {
        int n = a.length;
        if (n != b.length)
            throw arraySameLengthException(n, b.length);
        for (int i = 0; i < n; i++)
            addInPlace(a[i], b[i]);
    }

    /**
     * Multiply two arrays of {@code double}, overwriting the first.  The arrays must be the
     * same size.
     *
     * @param   a       the first array.
     * @param   b       the second array
     * @throws  IllegalArgumentException if the arrays are of different lengths
     */
    public static void multiplyInPlace(double[] a, double[] b) {
        int n = a.length;
        if (n != b.length)
            throw arraySameLengthException(n, b.length);
        for (int i = 0; i < n; i++)
            a[i] *= b[i];
    }

    private static IllegalArgumentException arraySameLengthException(int a, int b) {
        return new IllegalArgumentException("Arrays must be same length (" + a + " != " + b +
                ')');
    }

    /**
     * Create a display representation of the network for debug output.
     *
     * @return  the display form
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Network[");
        int i = 0;
        for (;;) {
            sb.append(getLayer(i).getSize());
            if (++i >= numLayers)
                break;
            sb.append(',');
        }
        sb.append(']');
        return sb.toString();
    }

    /**
     * Configure training (use {@link Trainer#go()} to start training operation).
     *
     * @return  a {@link Trainer} object
     */
    public Trainer train() {
        return new Trainer();
    }

    /**
     * Configure training with a specified {@link TrainingDataSource} (use {@link Trainer#go()}
     * to start training operation).
     *
     * @param   trainingData    the training data
     * @return  a {@link Trainer} object
     */
    public Trainer train(TrainingDataSource trainingData) {
        return train().trainingData(trainingData);
    }

    /**
     * Inner class to provide "fluent" interface for network training operations.
     */
    public class Trainer {

        private TrainingDataSource trainingData;
        private TrainingDataSource testData;
        private int epochs;
        private int miniBatchSize;
        private double eta;
        private Random random;

        public Trainer() {
            trainingData = null;
            testData = null;
            epochs = 30;
            miniBatchSize = 10;
            eta = 3.0;
            random = null;
        }

        public Trainer trainingData(TrainingDataSource trainingData) {
            this.trainingData = trainingData;
            return this;
        }

        public Trainer testData(TrainingDataSource testData) {
            this.testData = testData;
            return this;
        }

        public Trainer epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Trainer miniBatchSize(int miniBatchSize) {
            this.miniBatchSize = miniBatchSize;
            return this;
        }

        public Trainer eta(double eta) {
            this.eta = eta;
            return this;
        }

        public Trainer random(Random random) {
            this.random = random;
            return this;
        }

        public void go() {
            stochasticGradientDescent(trainingData, epochs, miniBatchSize, eta, random,
                    testData);
        }

    }

}
