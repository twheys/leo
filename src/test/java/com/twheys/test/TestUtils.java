package com.twheys.test;

import org.la4j.Matrix;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.stream.Stream;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public class TestUtils {

    public static Matrix[] getTestParameters(Path[] paths) {

        final Matrix[] parameters = new Matrix[paths.length];
        for (int i = 0; i < paths.length; i++) {
            try (final BufferedReader reader = Files.newBufferedReader(paths[i])) {
                parameters[i] = Matrix.from2DArray(streamData(reader).toArray(double[][]::new));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return parameters;
    }

    public static Stream<double[]> streamData(BufferedReader reader) {
        return reader.lines()
                .map(line -> Arrays.stream(line.split(","))
                        .mapToDouble(Double::valueOf)
                        .toArray());
    }

    public static double[] convertLabel(int value, int size) {
        final double[] labelArray = new double[size];
        labelArray[value % size] = 1;
        return labelArray;
    }

    public static int convertLabel(double[] predict) {
        double max = -1;
        int index = -1;
        for (int i = 0; i < predict.length; i++) {
            if (max < predict[i]) {
                max = predict[i];
                index = i;
            }
        }
        return index;
    }
}
