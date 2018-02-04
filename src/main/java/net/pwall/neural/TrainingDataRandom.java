/*
 * @(#) TrainingDataRandom.java
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
 * Default comment for {@code TrainingDataRandom}.
 *
 * @author  Peter Wall
 */
public class TrainingDataRandom implements TrainingDataSource {

    private TrainingDataSource source;
    private int[] index;

    /**
     * Construct a {@code TrainingDataRandom} from an original {@link TrainingDataSource}.
     *
     * @param   source  the {@link TrainingDataSource}
     */
    public TrainingDataRandom(TrainingDataSource source) {
        this.source = Objects.requireNonNull(source);
        int length = source.getSize();
        index = new int[length];
        for (int i = 0; i < length; i++)
            index[i] = i;
    }

    /**
     * Randomise the index - is there a better way of doing this?
     *
     * @param   r       a {@link Random}
     */
    public void randomise(Random r) {
        int i = getSize();
        while (i > 1) {
            int x = r.nextInt(i);
            int previous = index[x];
            System.arraycopy(index, x + 1, index, x, i - x - 1);
            index[--i] = previous;
        }
    }

    @Override
    public TrainingData getItem(int i) {
        return source.getItem(index[i]);
    }

    @Override
    public int getSize() {
        return index.length;
    }

}
