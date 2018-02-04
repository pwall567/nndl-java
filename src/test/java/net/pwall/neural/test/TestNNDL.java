/*
 * @(#) TestNNDL.java
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

import java.util.Random;

import net.pwall.neural.Network;
import net.pwall.neural.TrainingDataSource;
import net.pwall.neural.TrainingDataSubset;
import net.pwall.neural.test.images.MNISTImageData;
import net.pwall.neural.test.images.MNISTLabelData;

/**
 * Initial test class.
 *
 * <p>To enable logging, add:</p>
 * <pre>
 *     -Djava.util.logging.config.file=./target/test-classes/logging.properties
 * </pre>
 * <p>to the {@code java} command that runs the class (assumes that the current directory is the
 * root directory of the project).  Or switch to any other logging library supported by Apache
 * Commons Logging.</p>
 *
 * @author  Peter Wall
 */
public class TestNNDL {

    // modify the following filenames if necessary:

    public static final String imageDataFilename =
            "../../Downloads/MNIST/train-images.idx3-ubyte";
    public static final String labelDataFilename =
            "../../Downloads/MNIST/train-labels.idx1-ubyte";

    protected static MNISTImageData imageData;
    protected static MNISTLabelData labelData;

    public static void main(String[] args) {
        try {
            imageData = new MNISTImageData(imageDataFilename);
            labelData = new MNISTLabelData(labelDataFilename);
            Random r = new Random();

            // initialise network with layer sizes as in the original example

            Network network = new Network(784, 30, 10);
            network.init(r);

            // create two subsets of training data (as explained in the book)

            TrainingDataSource tds = new InputDataSource(imageData, labelData);
            TrainingDataSource trainingData = new TrainingDataSubset(tds, 0, 50000);
            TrainingDataSource testData = new TrainingDataSubset(tds, 50000, 10000);

            // run stochastic gradient descent with parameters in the example

            network.stochasticGradientDescent(trainingData, 30, 10, 3.0, r, testData);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

}
