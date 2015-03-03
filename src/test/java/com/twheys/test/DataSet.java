package com.twheys.test;

/**
 * @author <a href="mailto:theys@runwaynine.com">Timothy Heys</a>
 */
public class DataSet {
    private final double[][] inputs;
    private final double[][] labels;

    public DataSet(double[][] inputs, double[][] labels) {
        assert inputs.length == labels.length;
        this.inputs = inputs;
        this.labels = labels;
    }

    public double[][] getInputs() {
        return inputs;
    }

    public double[][] getLabels() {
        return labels;
    }

    public int size() {
        return inputs.length;
    }
}
