package org.nd4j.finitedifferences;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;

import java.util.Arrays;
import java.util.Collections;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;

public class TwoPointApproximationTest {


    @Test
    public void testDifference() {
        double spread = 1.0;
        double minDist = 1.0;
        INDArray xv = Nd4j.linspace(0,spread * 3,300);
        INDArray yv = Nd4j.create(xv.shape());
        //INDArray xvLtMinDist = BooleanIndexing.chooseFrom(new INDArray[]{xv}, Arrays.asList(minDist), Collections.emptyList(),new LessThan());
        INDArray xvLtMinDist = xv.lt(minDist);

        yv.put(xvLtMinDist,Nd4j.scalar(1.0));
        INDArray xvGteMinDist = BooleanIndexing.chooseFrom(new INDArray[]{xv}, Arrays.asList(minDist), Collections.emptyList(),new GreaterThanOrEqual());
        INDArray xvGteMinDistMinusMinDist = xvGteMinDist.sub(minDist);
        INDArray neg = xvGteMinDistMinusMinDist.neg();
        INDArray divSpread = neg.div(spread);

        INDArray toPut = exp(divSpread);

        yv.put(xvGteMinDist,toPut);
        TwoPointApproximation
                .approximateDerivative(indArray -> pow(indArray.mul(spread),2 * minDist)
                                .addi(1).rdivi(1.0),xv,null,yv,
                        Nd4j.create(new double[] {Double.NEGATIVE_INFINITY
                                ,Double.POSITIVE_INFINITY}));
    }
}
