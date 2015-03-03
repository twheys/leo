package com.twheys.ml;

import org.la4j.Matrix;

/**
 * @author <a href="mailto:theys@runwaynine.com">Timothy Heys</a>
 */
public interface ICostFunction {
    Matrix feed(Matrix inputs, Matrix[] theta);

    double cost(Matrix y, double alpha);

    double cost(Matrix y, Matrix[] theta, double alpha, double lambda);

    Matrix[] gradient(Matrix[] theta, Matrix y, double alpha);

    Matrix[] gradient(Matrix[] theta, Matrix y, double alpha, double lambda);
}
