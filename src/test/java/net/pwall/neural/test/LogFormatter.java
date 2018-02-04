/*
 * @(#) LogFormatter.java
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

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Calendar;
import java.util.logging.Formatter;
import java.util.logging.LogRecord;

/**
 * Simple log formatter for {@code java.util.logging} that produces fairly concise output.
 *
 * @author      Peter Wall
 */
public class LogFormatter extends Formatter {

    private String lineTerminator = System.getProperty("line.separator", "\n");

    @Override
    public String format(LogRecord record) {
        StringBuilder sb = new StringBuilder();
        Calendar cal = Calendar.getInstance();
        cal.setTimeInMillis(record.getMillis());
// Uncomment the following if you like:
//        sb.append(cal.get(Calendar.YEAR));
//        sb.append('-');
//        append2Digit(sb, cal.get(Calendar.MONTH) + 1);
//        sb.append('-');
//        append2Digit(sb, cal.get(Calendar.DAY_OF_MONTH));
//        sb.append(' ');
        append2Digit(sb, cal.get(Calendar.HOUR_OF_DAY));
        sb.append(':');
        append2Digit(sb, cal.get(Calendar.MINUTE));
        sb.append(':');
        append2Digit(sb, cal.get(Calendar.SECOND));
        sb.append(' ');
        sb.append(record.getLevel());
        sb.append(' ');
        sb.append(record.getMessage());
        sb.append(lineTerminator);
        Throwable thrown = record.getThrown();
        if (thrown != null) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            pw.print(sb.toString());
            thrown.printStackTrace(pw);
            return sw.toString();
        }
        return sb.toString();
    }

    private static void append2Digit(StringBuilder sb, int i) {
        sb.append((char)((i / 10) % 10 + '0'));
        sb.append((char)(i % 10 + '0'));
    }

}
