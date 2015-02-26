package com.twheys.ai;

import com.twheys.ml.IGradientFunction;
import org.la4j.Matrix;

import java.util.Random;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public class LA4JNeuralNetwork implements ArtificialNeuralNetwork {

    public static LA4JNeuralNetwork create(IGradientFunction f, double[][][] weights) {
        assert null != weights && 2 < weights.length;

        final Matrix[] weightMatricies = new Matrix[weights.length];
        for (int i = 0; i < weights.length; i++) {
            weightMatricies[i] = Matrix.from2DArray(weights[i]);
        }

        return new LA4JNeuralNetwork(weightMatricies, f);
    }

    public static LA4JNeuralNetwork create(IGradientFunction f, int... layerSizes) {
        assert null != layerSizes && 2 < layerSizes.length;
        final Random random = new Random();

        final Matrix[] weights = new Matrix[layerSizes.length - 1];
        for (int i = 0; i < layerSizes.length - 1; i++) {
            weights[i] = Matrix.random(layerSizes[i + 1], layerSizes[i] + 1, random)
                    .multiply(2)
                    .subtract(1);
        }

        return new LA4JNeuralNetwork(weights, f);
    }

    private final Matrix[] weights;
    private final IGradientFunction f;

    public LA4JNeuralNetwork(final Matrix[] weights, IGradientFunction f) {
        this.weights = weights;
        this.f = f;
    }

    @Override
    public double train(double[][] inputs, double[][] trainingSet) {
        return train(inputs, trainingSet, 50);
    }

    @Override
    public double train(double[][] inputs, double[][] trainingSet, int maxIterations) {
        return train(Matrix.from2DArray(inputs), Matrix.from2DArray(trainingSet), maxIterations);
    }

    public double train(Matrix inputs, Matrix trainingSet, int maxIterations) {
        assert inputs.columns() == weights[0].columns() - 1;

        final double alpha = 2.5;
        final double lambda = 10;

        double lastCost = 0;
        double averageCost = 0;
        double deltaCost = 1;

        System.out.println("--- Training ANN ---");
        for (int i = 0; i < maxIterations && 0.00001 < Math.abs(deltaCost); i++) {
            f.feed(inputs, weights);

            final double cost = f.cost(trainingSet, weights, alpha, lambda);
            final Matrix[] gradients = f.gradient(weights, trainingSet, alpha, lambda);

            averageCost = (averageCost * i + cost) / (i + 1);
            deltaCost = cost - lastCost;
            lastCost = cost;

            System.out.printf("[%d] Cost:%f Δ:%f Avg:%f%n", i + 1, cost, deltaCost, averageCost);
            for (int l = 0; l < weights.length; l++) {
                weights[l] = weights[l].subtract(gradients[l]);
            }
        }
        return lastCost;
    }

    @Override
    public double[] predict(double[] inputs) {
        final Matrix h = f.feed(Matrix.from1DArray(1, inputs.length, inputs), weights);
        final double[] outputs = new double[h.columns()];
        for (int i = 0; i < h.columns(); i++) {
            outputs[i] = h.get(0, i);
        }

        return outputs;
    }
}
