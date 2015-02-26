package com.twheys;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.iterator.MatrixIterator;
import org.la4j.iterator.VectorIterator;

import java.util.function.Function;

/**
 * @author <a href="mailto:twheys@gmail.com">Timothy Heys</a>
 */
public class MatrixUtils {
    public static Vector apply(Vector input, Function<Double, Double> f) {
        VectorIterator it = input.iterator();
        Vector result = input.blank();

        while (it.hasNext()) {
            double x = it.next();
            int i = it.index();
            result.set(i, f.apply(x));
        }

        return result;
    }

    public static Matrix apply(Matrix input, Function<Double, Double> f) {
        Matrix result = input.blank();
        MatrixIterator it = input.iterator();

        while (it.hasNext()) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            result.set(i, j, f.apply(x));
        }

        return result;
    }

    public static Vector log(Vector input) {
        return MatrixUtils.apply(input, Math::log);
    }


    public static Matrix log(Matrix input) {
        return MatrixUtils.apply(input, Math::log);
    }

    public static Vector pow(Vector input, double exp) {
        return MatrixUtils.apply(input, val -> Math.pow(val, exp));
    }

    public static Matrix pow(Matrix input, double exp) {
        return MatrixUtils.apply(input, val -> Math.pow(val, exp));
    }

    /**
     * Adds one row to matrix.
     *
     * @param i the row index
     * @return matrix with row.
     */
    public static Matrix addRow(Matrix to, int i, Vector row) {
        if (i >= to.rows() || i < 0) {
            throw new IndexOutOfBoundsException("Illegal row number, must be 0.." + (to.rows() - 1));
        }

        Matrix result = to.blankOfShape(to.rows() + 1, to.columns());

        for (int ii = 0; ii < i; ii++) {
            result.setRow(ii, to.getRow(ii));
        }

        result.setRow(i, row);

        for (int ii = i; ii < to.rows(); ii++) {
            result.setRow(ii + 1, to.getRow(ii));
        }

        return result;
    }

    /**
     * Adds one column to matrix.
     *
     * @param j the column index
     * @return matrix with column.
     */
    public static Matrix addColumn(Matrix to, int j, Vector column) {
        if (j >= to.columns() || j < 0) {
            throw new IndexOutOfBoundsException("Illegal row number, must be 0.." + (to.columns() - 1));
        }

        Matrix result = to.blankOfShape(to.rows(), to.columns() + 1);

        for (int jj = 0; jj < j; jj++) {
            result.setColumn(jj, to.getColumn(jj));
        }

        result.setColumn(j, column);

        for (int jj = j; jj < to.columns(); jj++) {
            result.setColumn(jj + 1, to.getColumn(jj));
        }

        return result;
    }
}
