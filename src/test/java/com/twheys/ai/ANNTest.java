package com.twheys.ai;

import com.twheys.ml.BackpropagationFunction;
import com.twheys.ml.IGradientFunction;
import com.twheys.ml.SigmoidFunction;
import com.twheys.test.AcceptanceTest;
import com.twheys.test.TestUtils;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.BufferedReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static com.twheys.test.TestUtils.convertLabel;
import static com.twheys.test.TestUtils.streamData;
import static org.junit.Assert.assertTrue;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
@Category(AcceptanceTest.class)
public class ANNTest {

    public static Path DATA_PATH;
    public static Path LABELS_PATH;

    @BeforeClass
    public static void setUp() throws URISyntaxException {
        DATA_PATH = Paths.get(ANNTest.class.getResource("/data/handwriting/images.csv").toURI());
        LABELS_PATH = Paths.get(ANNTest.class.getResource("/data/handwriting/labels.csv").toURI());
    }

    @Test
    public void testHandwritingSamplesWithSigmoid() throws IOException {
        testANN(new BackpropagationFunction(new SigmoidFunction()));
    }

    private void testANN(IGradientFunction f) throws IOException {
        final int labelSize = 10;
        ArtificialNeuralNetwork ann = LA4JNeuralNetwork.create(f, 400, 25, labelSize);

        try (final BufferedReader dataReader = Files.newBufferedReader(DATA_PATH);
             final BufferedReader labelReader = Files.newBufferedReader(LABELS_PATH)) {
            double[][] data = streamData(dataReader).toArray(size -> new double[size][]);
            double[][] labels = labelReader.lines()
                    .map(val -> TestUtils.convertLabel(Integer.valueOf(val), labelSize))
                    .toArray(size -> new double[size][]);

            ann.train(data, labels, 50);

            int totalCorrect = 0;
            for (int i = 0; i < data.length; i++) {
                int prediction = convertLabel(ann.predict(data[i]));
                int label = convertLabel(labels[i]);

                if (prediction == label)
                    totalCorrect++;
            }
            final double accuracy = 100d * ((double) totalCorrect) / data.length;
            System.out.printf("Score: %d/%d %f%%%n", totalCorrect, data.length, accuracy);

            assertTrue(85 < accuracy);
        }
    }
}
