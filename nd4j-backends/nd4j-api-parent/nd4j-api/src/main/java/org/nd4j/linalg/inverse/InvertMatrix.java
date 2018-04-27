package org.nd4j.linalg.inverse;

import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsoncccc on 11/30/15.
 */
public class InvertMatrix {


    /**
     * Inverts a matrix
     * @param arr the array to invert
     * @return the inverted matrix
     */
    public static INDArray invert(INDArray arr, boolean inPlace) {
        if (!arr.isSquare()) {
            throw new IllegalArgumentException("invalid array: must be square matrix");
        }

        //FIX ME: Please
       /* int[] IPIV = new int[arr.length() + 1];
        int LWORK = arr.length() * arr.length();
        INDArray WORK = Nd4j.create(new double[LWORK]);
        INDArray inverse = inPlace ? arr : arr.dup();
        Nd4j.getBlasWrapper().lapack().getrf(arr);
        Nd4j.getBlasWrapper().lapack().getri(arr.size(0),inverse,arr.size(0),IPIV,WORK,LWORK,0);*/

        RealMatrix rm = CheckUtil.convertToApacheMatrix(arr);
        RealMatrix rmInverse = new LUDecomposition(rm).getSolver().getInverse();


        INDArray inverse = CheckUtil.convertFromApacheMatrix(rmInverse);
        if (inPlace)
            arr.assign(inverse);
        return inverse;

    }

    /**
     * Calculates pseudo inverse of a matrix using QR decomposition
     * @param arr the array to invert
     * @return the pseudo inverted matrix
     */
    public static INDArray pinvert(INDArray arr, boolean inPlace) {

        // TODO : do it natively instead of relying on commons-maths

        RealMatrix realMatrix = CheckUtil.convertToApacheMatrix(arr);
        QRDecomposition decomposition = new QRDecomposition(realMatrix, 0);
        DecompositionSolver solver = decomposition.getSolver();

        if (!solver.isNonSingular()) {
            throw new IllegalArgumentException("invalid array: must be singular matrix");
        }

        RealMatrix pinvRM = solver.getInverse();

        INDArray pseudoInverse = CheckUtil.convertFromApacheMatrix(pinvRM);

        if (inPlace)
            arr.assign(pseudoInverse);
        return pseudoInverse;

    }

}
