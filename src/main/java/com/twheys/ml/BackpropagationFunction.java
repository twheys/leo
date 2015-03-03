package com.twheys.ml;

import com.twheys.MatrixUtils;
import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.Vectors;
import org.la4j.vector.functor.VectorAccumulator;

import java.util.Arrays;

import static com.twheys.MatrixUtils.*;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public class BackpropagationFunction implements ICostFunction {
    private final ITransferFunction t;
    private final VectorAccumulator sumAccumulator = Vectors.asSumAccumulator(0.0);

    private Matrix[] activations = null;

    void setActivations(Matrix[] activations) {
        this.activations = activations;
    }

    public BackpropagationFunction(ITransferFunction transferFunction) {
        this.t = transferFunction;
    }

    //    %% Feed Forward
    //    a1 = [ones(m,1) X];
    //    z2 = (Theta1 * a1')';
    //    a2 = [ones(m,1) sigmoid(z2)];
    //    z3 = (Theta2 * a2')';
    //    a3 = sigmoid(z3);
    @Override
    public Matrix feed(Matrix inputs, Matrix[] theta) {
        final Matrix[] activations = new Matrix[theta.length + 1];
        final Vector ones = Vector.constant(inputs.rows(), 1d);

        int l = 0;
        activations[l] = addColumn(inputs, 0, ones);
        for (l++; l < activations.length; l++) {
            final Matrix z = theta[l - 1].multiply(activations[l - 1].transpose()).transpose();
            if (l < theta.length) {
                activations[l] = addColumn(t.transfer(z), 0, ones);
            } else {
                activations[l] = t.transfer(z);
            }
        }

        this.activations = activations;
        return activations[activations.length - 1];
    }

    //    %% Cost function
    //    J = mean(sum(-y .* log(h) - (1 - y) .* log(1 - h), 2))
    @Override
    public double cost(Matrix y, double alpha) {
        final Matrix h = activations[activations.length - 1];
        assert y.rows() == h.rows();
        assert y.columns() == h.columns();

        final double m = y.rows();
        return alpha / m * Vector.fromArray(y.multiply(-1)
                .hadamardProduct(log(h))
                .subtract(y.multiply(-1)
                        .add(1)
                        .hadamardProduct(log(h.multiply(-1)
                                .add(1))))
                .foldColumns(sumAccumulator))
                .sum();
    }

    //    J += lambda / (2 * m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
    @Override
    public double cost(Matrix y, Matrix[] theta, double alpha, double lambda) {
        final double m = y.rows();
        return cost(y, alpha) + lambda / (2. * m) * Arrays.stream(theta)
                .mapToDouble(matrix -> pow(matrix.removeColumn(0), 2d)
                        .sum())
                .sum();
    }

    //    %% Back Propagation
    //    Theta1_grad = 1 / m .* (d2 * a1);
    //    Theta2_grad = 1 / m .* (d3 * a2);
    @Override
    public Matrix[] gradient(Matrix[] theta, Matrix y, double alpha) {
        final int outputLayer = activations.length - 1;
        assert y.rows() == activations[outputLayer].rows();
        assert y.columns() == activations[outputLayer].columns();

        final Matrix[] derivatives = derivative(activations, y, theta);

        final Matrix[] gradients = new Matrix[theta.length];
        final double m = y.rows();
        for (int l = 0; gradients.length > l; l++) {
            gradients[l] = derivatives[l + 1]
                    .multiply(activations[l])
                    .multiply(alpha / m);
        }

        return gradients;
    }

    //    %% Back Propagation Regularization
    //    Theta1_grad += [zeros(rows(Theta1),1) lambda / m * Theta1(:,2:end)];
    //    Theta2_grad += [zeros(rows(Theta2),1) lambda / m * Theta2(:,2:end)];
    @Override
    public Matrix[] gradient(Matrix[] theta, Matrix y, double alpha, double lambda) {
        final Matrix[] gradients = this.gradient(theta, y, alpha);
        final Matrix[] regularizedGradients = new Matrix[gradients.length];
        final double m = y.rows();
        for (int l = 0; gradients.length > l; l++) {
            final Vector zeros = Vector.zero(gradients[l].rows());
            regularizedGradients[l] = gradients[l]
                    .add(MatrixUtils.addColumn(theta[l]
                            .removeColumn(0)
                            .multiply(lambda / m), 0, zeros));
        }
        return regularizedGradients;
    }

    //    %% Back Propagation - Derivatives of activations
    //    d3 = (a3 - yBool)';
    //    d2 = (Theta2' * d3)(2:end,:) .* sigmoidGradient(z2');
    private Matrix[] derivative(Matrix[] activations, Matrix y, Matrix[] theta) {
        final int outputLayer = activations.length;
        final Matrix[] derivatives = new Matrix[outputLayer];
        derivatives[outputLayer - 1] = activations[outputLayer - 1].subtract(y).transpose();
        for (int l = outputLayer - 2; 0 < l; l--) {
            derivatives[l] = theta[l]
                    .transpose()
                    .multiply(derivatives[l + 1])
                    .removeRow(0)
                    .hadamardProduct(t.derivative(activations[l]
                            .transpose()
                            .removeRow(0)));
        }
        return derivatives;
    }
}
