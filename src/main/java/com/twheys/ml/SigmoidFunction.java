package com.twheys.ml;

import com.twheys.ml.ITransferFunction;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public class SigmoidFunction implements ITransferFunction {
    /**
     * f(z)=1/(1+exp(−z))
     */
    @Override
    public double transfer(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * f′(z)=f(z)(1−f(z))
     */
    @Override
    public double derivative(double x) {
        return x * (1.0 - x);
    }
}
