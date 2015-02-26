package com.twheys.ml;

import static java.lang.Math.max;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public class RectifiedLinearFunction implements ITransferFunction {
    @Override
    public double transfer(double x) {
        return max(0, x);
    }

    @Override
    public double derivative(double x) {
        return 0 < x ? 1 : 0;
    }
}
