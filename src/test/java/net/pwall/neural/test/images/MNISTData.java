/*
 * @(#) MNISTLabelData.java
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

import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * MNIST Data parent class.
 */
public abstract class MNISTData {

    private String filename;
    private InputStream in;

    public MNISTData(String filename, int magicNumber) throws IOException {
        this.filename = filename;
        in = new FileInputStream(filename);
        if (read32() != magicNumber)
            throw new IOException("Incorrect magic number");
    }

    public String getFilename() {
        return filename;
    }

    protected int read8() throws IOException {
        int result = in.read();
        if (result < 0)
            throw new EOFException("Unexpected EOF");
        return result;
    }

    protected int read32() throws IOException {
        int result = 0;
        for (int i = 4; i > 0; i--)
            result = (result << 8) | (read8() & 0xFF);
        return result;
    }

    protected void readArray(byte[] buf, int len) throws IOException {
        int offset = 0;
        while (len > 0) {
            int n = in.read(buf, offset, len);
            if (n < 0)
                throw new EOFException("Unexpected EOF");
            offset += n;
            len -= n;
        }
    }

}
