package org.nd4j.finitedifferences;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * Simple 2 point finite difference approximation
 * to compute the partial derivatives wrt the 2 given  points
 * based on:
 * https://github.com/apache/commons-math/blob/master/src/main/java/org/apache/commons/math4/analysis/interpolation/BicubicInterpolator.java
 *
 * @author Adam Gibson
 */
public class TwoPointApproximation {


    /**
     * Adjust final scheme to presence of bounds
     * @param x the point to estimate the derivative
     * @param h the finite difference steps
     * @param numSteps Number of h steps in 1 direction
     *                 to implement finite difference scheme.
     *
     * @param lowerBound Lower bounds for independent variable variable
     * @param upperBound
     * @return
     */
    public static INDArray adjustSchemeToBounds(INDArray x,INDArray h,int numSteps,INDArray lowerBound,INDArray upperBound) {
        h = abs(h);
        INDArray oneSided = Nd4j.zerosLike(h);

        /**
         * COME BACK TO THIS
         *     if np.all((lb == -np.inf) & (ub == np.inf)):
         return h, use_one_sided

         */

        INDArray hTotal = h.mul(numSteps);
        INDArray hAdjusted = h.dup();
        INDArray lowerDist = x.sub(lowerBound);
        INDArray upperBound2 = upperBound.sub(x);

        INDArray central = and(greaterThanOrEqual(lowerDist,hTotal),greaterThanOrEqual(upperBound2,hTotal));
        INDArray forward = and(greaterThanOrEqual(upperBound,lowerDist),not(central));
        int[] fowardIndices = forward.data().asInt();
        //hAdjusted.put(new INDArrayIndex[]{
        //         new SpecifiedIndex(fowardIndices)},min(h.get));
        return hTotal;
    }

    public static INDArray computeAbsoluteStep(INDArray x) {
        INDArray relStep = pow(Nd4j.scalar(Nd4j.EPS_THRESHOLD),0.5);
        return computeAbsoluteStep(relStep,x);
    }

    public static INDArray computeAbsoluteStep(INDArray relStep,INDArray x) {
        INDArray signX0 = x.gte(0).muli(2).subi(1);
        return relStep.mul(signX0).muli(max(abs(x),1.0));
    }

    public static INDArray approximateDerivative(Function<INDArray,INDArray> f,INDArray x, INDArray relStep,INDArray bounds)  {
        INDArray h = computeAbsoluteStep(relStep,x);
        return h;

    }

}
