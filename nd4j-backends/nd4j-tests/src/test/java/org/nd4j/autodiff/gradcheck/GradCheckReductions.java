package org.nd4j.autodiff.gradcheck;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertTrue;

@Slf4j
public class GradCheckReductions {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }


    @Test
    public void testReductionGradientsFullArray(){

        for (int i = 0; i < 7; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 10;
            SDVariable input = sd.var("in", new int[]{-1, nOut});
            SDVariable label = sd.var("label", new int[]{-1, nOut});

            SDVariable diff = input.sub(label);
            SDVariable sqDiff = diff.mul(diff);
            SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);

            SDVariable loss;
            String name;
            switch (i) {
                case 0:
                    loss = sd.mean("loss", msePerEx, 0);
                    name = "mean";
                    break;
                case 1:
                    loss = sd.sum("loss", msePerEx, 0);
                    name = "sum";
                    break;
                case 2:
                    loss = sd.standardDeviation("loss", msePerEx, true, 0);
                    name = "stdev";
                    break;
                case 3:
                    loss = sd.min("loss", msePerEx, 0);
                    name = "min";
                    break;
                case 4:
                    loss = sd.max("loss", msePerEx, 0);
                    name = "max";
                    break;
                case 5:
                    loss = sd.variance("loss", msePerEx, true, 0);
                    name = "variance";
                    break;
                case 6:
                    loss = sd.prod("loss", msePerEx, 0);
                    name = "prod";
                    break;
                default:
                    throw new RuntimeException();
            }


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            INDArray inputArr = Nd4j.rand(minibatch, nOut);
            INDArray labelArr = Nd4j.rand(minibatch, nOut);

            sd.associateArrayWithVariable(inputArr, input);
            sd.associateArrayWithVariable(labelArr, label);

            boolean ok = GradCheckUtil.checkGradients(sd);


            assertTrue(msg, ok);
        }


    }
}
