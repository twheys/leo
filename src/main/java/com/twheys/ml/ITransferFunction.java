package com.twheys.ml;

import org.la4j.Matrix;
import org.la4j.Vector;

import static com.twheys.MatrixUtils.apply;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public interface ITransferFunction {
    double transfer(double x);

    default Vector transfer(Vector x) {
        return apply(x, this::transfer);
    }

    default Matrix transfer(Matrix x) {
        return apply(x, this::transfer);
    }

    double derivative(double x);

    default Vector derivative(Vector x) {
        return apply(x, this::derivative);
    }

    default Matrix derivative(Matrix x) {
        return apply(x, this::derivative);
    }
}
