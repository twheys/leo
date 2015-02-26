package com.twheys.ml;

import static java.lang.Math.pow;
import static java.lang.Math.tanh;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public class TanhFunction implements ITransferFunction {
    @Override
    public double transfer(double x) {
        return tanh(x);
    }

    @Override
    public double derivative(double x) {
        return 1 - pow(tanh(x), 2d);
    }
}
