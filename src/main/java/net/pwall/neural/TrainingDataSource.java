/*
 * @(#) TrainingDataSource.java
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

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Spliterator;
import java.util.function.Consumer;

/**
 * Source of training data for Stochastic Gradient Descent.  To implement this interface two
 * methods must be provided: {@link #getSize()} the get the number of entries in this training
 * data set, and {@link #getItem(int)} to get an individual {@link TrainingData} item (inputs
 * and expected outputs).
 *
 * @author Peter Wall
 */
public interface TrainingDataSource extends Iterable<TrainingData> {

    /**
     * Get a {@link TrainingData} item.
     *
     * @param   index   the index of the item
     * @return  the {@link TrainingData} item
     */
    TrainingData getItem(int index);

    /**
     * Get the size of the training data (number of entries).
     *
     * @return  the number of entries
     */
    int getSize();

    /**
     * Get an {@link Iterator} for this {@code TrainingDataSource}.
     *
     * @return  the {@link Iterator}
     */
    @Override
    default Iterator<TrainingData> iterator() {
        return new TrainingDataIterator(this);
    }

    /**
     * Get a {@link Spliterator} for this {@code TrainingDataSource}.
     *
     * @return  the {@link Spliterator}
     */
    @Override
    default Spliterator<TrainingData> spliterator() {
        return new TrainingDataSpliterator(this);
    }

    static class TrainingDataIterator implements Iterator<TrainingData> {

        private TrainingDataSource source;
        private int index;

        public TrainingDataIterator(TrainingDataSource source) {
            this.source = source;
            index = 0;
        }

        @Override
        public boolean hasNext() {
            return index < source.getSize();
        }

        @Override
        public TrainingData next() {
            if (!hasNext())
                throw new NoSuchElementException();
            return source.getItem(index++);
        }

    }

    /**
     * This implementation of the {@link Spliterator} interface is provided to allow future
     * implementations to make use of parallel map/reduce techniques.
     */
    static class TrainingDataSpliterator implements Spliterator<TrainingData> {

        private TrainingDataSource source;
        private int index;
        private int limit;

        public TrainingDataSpliterator(TrainingDataSource source) {
            this.source = source;
            index = 0;
            limit = source.getSize();
        }

        private TrainingDataSpliterator(TrainingDataSource source, int index, int limit) {
            this.source = source;
            this.index = index;
            this.limit = limit;
        }

        @Override
        public boolean tryAdvance(Consumer<? super TrainingData> action) {
            Objects.requireNonNull(action);
            if (index < limit) {
                action.accept(source.getItem(index++));
                return true;
            }
            return false;
        }

        @Override
        public Spliterator<TrainingData> trySplit() {
            int size = limit - index;
            if (size < 2)
                return null;
            int oldIndex = index;
            index += size >>> 1;
            return new TrainingDataSpliterator(source, oldIndex, index);
        }

        @Override
        public long estimateSize() {
            return limit - index;
        }

        @Override
        public int characteristics() {
            return ORDERED | SIZED | IMMUTABLE | SUBSIZED;
        }

        @Override
        public void forEachRemaining(Consumer<? super TrainingData> action) {
            Objects.requireNonNull(action);
            while (index < limit)
                action.accept(source.getItem(index++));
        }

    }

}
