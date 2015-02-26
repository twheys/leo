package com.twheys.ml;

import com.twheys.test.TestUtils;
import org.junit.BeforeClass;
import org.junit.Test;
import org.la4j.Matrix;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public class GradientDescentTest {

    private static Path[] testParametersPaths;

    private static final Matrix y = Matrix.from2DArray(new double[][]{
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    });
    private static final Matrix h = Matrix.from2DArray(new double[][]{
            {1.1266e-04, 1.7413e-03, 2.5270e-03, 1.8403e-05, 9.3626e-03, 3.9927e-03, 5.5152e-03, 4.0147e-04, 6.4807e-03, 9.9573e-01},
            {9.3990e-01, 5.4025e-03, 2.9890e-02, 3.1662e-04, 2.0517e-03, 9.0675e-04, 2.7390e-02, 4.9149e-03, 2.5644e-02, 1.4278e-05}
    });

    @BeforeClass
    public static void setUp() throws URISyntaxException {
        testParametersPaths = new Path[]{
                Paths.get(TestUtils.class.getResource("/data/handwriting/theta1.csv").toURI()),
                Paths.get(TestUtils.class.getResource("/data/handwriting/theta1.csv").toURI())
        };
    }

    private BackpropagationFunction gd = new BackpropagationFunction(null);

    @Test
    public void testCostFunction() {
        gd.setActivations(new Matrix[]{h});
        assertEquals(0.097119, gd.cost(y, 1d), 0.0001);
    }

    @Test
    public void testCostFunctionRegularization() {
        Matrix[] theta = TestUtils.getTestParameters(testParametersPaths);
        gd.setActivations(new Matrix[]{h});
        assertEquals(2.06004, gd.cost(y, theta, 1d, 0.01), 0.0001);
    }
}
