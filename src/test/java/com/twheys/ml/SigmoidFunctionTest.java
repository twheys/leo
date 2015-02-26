package com.twheys.ml;

import org.junit.Test;
import org.la4j.Matrix;

import static org.junit.Assert.assertTrue;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public class SigmoidFunctionTest {

    final SigmoidFunction sigmoid = new SigmoidFunction();

    @Test
    public void testSigmoidFunction() {
        Matrix inputs = Matrix.from1DArray(1, 5, new double[]{-1, -0.5, 0, 0.5, 1});
        Matrix expectedOutputs = Matrix.from1DArray(1, 5, new double[]{0.26894, 0.37754, 0.50000, 0.62246, 0.73106});

        assertTrue(expectedOutputs.equals(sigmoid.transfer(inputs), 0.0001d));
    }

    @Test
    public void testSigmoidDerivative() {
        Matrix inputs = Matrix.from1DArray(1, 5, new double[]{-1, -0.5, 0, 0.5, 1});
        Matrix expectedOutputs = Matrix.from1DArray(1, 5, new double[]{0.196612, 0.235004, 0.250000, 0.235004, 0.196612});

        assertTrue(expectedOutputs.equals(sigmoid.derivative(sigmoid.transfer(inputs)), 0.0001d));
    }
}
