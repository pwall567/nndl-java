# nndl-java

Java Implementation of Neural Networks and Deep Learning

## Background

This project is an attempt to implement in Java the example code from the online book
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by
[Michael Nielsen](http://michaelnielsen.org/).

The book is an excellent exposition of the theory and the mathematics of neural networks, and it
includes complete examples in Python of code to recognise hand-written digits from image data.
Simple character recognition is a good starting point for the understanding of neural networks,
and it is possible to see that with the right choice of inputs the same technique can be applied
to a wide range of problems.

I am not very familiar with Python, so I decided to produce a Java version of the example code.
I had three motivations:

1.  I hoped that translating the code would give me a deeper understanding of the techniques
    used.
2.  A Java version would give me a library I could use for my own experimentation.
3.  I assumed that a type-safe language would give better performance than an untyped language.

On the last point, I failed to notice that the Python examples make extensive use of the
`numpy` library, and that library is written in C.  Still, I did manage to get some degree of
performance improvement, just not as much as I hoped.

I had also hoped to be able to make use of stream (map/reduce) capabilities of Java 8 to get
further performance advantages, but it seems likely the the overhead of invoking these features
would outweigh any potential performance improvements.  On the other hand, the use of standard
numerical libraries in Python allows for the possibility of employing GPU-based array
processing, further undermining my ideas of performance gains.

The project is a work in progress, and so far covers only chapters 1 and 2 of the book.  I plan
to add more as time permits.

## Requirements

The code is structured as a [Maven](https://maven.apache.org/) project.  I use
[Eclipse](http://www.eclipse.org/) but the code should work in any IDE, or none.

The data used is [The MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
and anyone wishing to try out the code should download the data from the page in this link.
