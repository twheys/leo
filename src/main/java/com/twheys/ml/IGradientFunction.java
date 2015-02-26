package com.twheys.ml;

import org.la4j.Matrix;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public interface IGradientFunction {

    Matrix feed(Matrix inputs, Matrix[] theta);

    /**
     * Cost function.
     *
     * @param y     The training set values as a logical array
     * @param alpha
     * @return
     */
    double cost(Matrix y, double alpha);

    /**
     * Regularized cost function.
     *
     * @param y
     * @param theta
     * @param alpha
     * @param lambda @return
     */
    double cost(Matrix y, Matrix[] theta, double alpha, double lambda);

    /**
     * @param theta
     * @param y
     * @param alpha
     * @return
     */
    Matrix[] gradient(Matrix[] theta, Matrix y, double alpha);

    /**
     * Regularized gradient function
     *
     * @param theta
     * @param y
     * @param alpha
     * @param lambda @return
     */
    Matrix[] gradient(Matrix[] theta, Matrix y, double alpha, double lambda);
}
