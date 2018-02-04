/*
 * @(#) TrainingDataSubset.java
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

/**
 * Default comment for {@code TrainingDataSubset}.
 *
 * @author  Peter Wall
 */
public class TrainingDataSubset implements TrainingDataSource {

    private TrainingDataSource source;
    private int start;
    private int length;

    /**
     * Construct a {@code TrainingDataSubset} from an original {@link TrainingDataSource}.
     *
     * @param   source  the {@link TrainingDataSource}
     */
    public TrainingDataSubset(TrainingDataSource source, int start, int length) {
        this.source = Objects.requireNonNull(source);
        if (start < 0 || length <= 0 || start + length > source.getSize())
            throw new IllegalArgumentException("start / length do not describe valid subset");
        this.start = start;
        this.length = length;
    }

    @Override
    public TrainingData getItem(int i) {
        if (i < 0 || i >= length)
            throw new IllegalArgumentException("index is not in range: " + i);
        return source.getItem(start + i);
    }

    @Override
    public int getSize() {
        return length;
    }

}
