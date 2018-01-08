package org.nd4j.autodiff.gradcheck;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.loss.LossFunctions;
import org.nd4j.autodiff.loss.LossInfo;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@Slf4j
public class GradCheckLoss {

    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testLossSimple2d(){
        //Test reductions: final and only function
        Nd4j.getRandom().setSeed(12345);

        for (int i = 0; i < 1; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 10;
            SDVariable input = sd.var("in", new int[]{-1, nOut});
            SDVariable labels = sd.var("labels", new int[]{-1, nOut});
//            SDVariable tempOnes = sd.one("ones", new int[]{minibatch, nOut});

            INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
            INDArray labelsArr = Nd4j.randn(minibatch, nOut).muli(100);

            LossInfo lossInfo;
            INDArray expOut;
            switch (i) {
                case 0:
                    lossInfo = LossFunctions.mse("out", input, labels, null, LossFunctions.Reduction.MEAN_BY_COUNT, 1);
                    inputArr.sub(labelsArr);
                    expOut = inputArr.sub(labelsArr);
                    expOut.muli(expOut);
                    expOut = expOut.mean(Integer.MAX_VALUE);
                    break;
                default:
                    throw new RuntimeException();
            }


            String msg = "test: " + i + " - " + lossInfo.getLossName();
            log.info("*** Starting test: " + msg);


            sd.associateArrayWithVariable(inputArr, input);
            sd.associateArrayWithVariable(labelsArr, labels);

//            System.out.println(sd.asFlatPrint());

            INDArray out = sd.execAndEndResult();

            assertEquals(msg, expOut, out);

            System.out.println("STARTING GRADIENT CHECK");
            boolean ok = GradCheckUtil.checkGradients(sd);

            assertTrue(msg, ok);
        }
    }

    @Test
    public void testSquare(){
        Nd4j.getRandom().setSeed(12345);

        int mb = 5;
        int nOut = 4;

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.rand(mb, nOut));
        SDVariable label = sd.var("label", Nd4j.rand(mb, nOut));
        SDVariable diff = in.sub(label);
        SDVariable sqDiff = sd.square(diff);

        INDArray expOut = in.getArr().sub(label.getArr());
        expOut.muli(expOut);

        System.out.println("About to exec");
        INDArray out = sd.execAndEndResult();   //JVM crash

        assertEquals(out, expOut);
    }

    @Test
    public void testOnesBroadcast(){
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.zero("in", new int[]{3,4});
        SDVariable one = sd.one("one", new int[]{1,1});

        SDVariable add = in.add(one);

        INDArray out = sd.execAndEndResult();
        assertEquals(Nd4j.ones(3,4), out);
    }

}
