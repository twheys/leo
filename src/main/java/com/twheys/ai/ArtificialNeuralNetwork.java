package com.twheys.ai;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public interface ArtificialNeuralNetwork {

    double train(double[][] inputs, double[][] trainingSet);

    double train(double[][] inputs, double[][] trainingSet, int maxIterations);

    double[] predict(double[] inputs);
}
