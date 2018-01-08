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
import org.nd4j.linalg.ops.transforms.Transforms;

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

        for (String fn : new String[]{"mse", "l1", "l2", "mcxent"}) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 10;
            SDVariable input = sd.var("in", new int[]{-1, nOut});
            SDVariable labels = sd.var("labels", new int[]{-1, nOut});

            INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
            INDArray labelsArr = Nd4j.randn(minibatch, nOut).muli(100);

            LossInfo lossInfo;
            INDArray expOut;
            switch (fn) {
                case "mse":
                    lossInfo = LossFunctions.mse("out", input, labels, null, LossFunctions.Reduction.MEAN_BY_COUNT, 1);
                    expOut = inputArr.sub(labelsArr);
                    expOut.muli(expOut);
                    expOut = expOut.mean(Integer.MAX_VALUE);
                    break;
                case "l1":
                    lossInfo = LossFunctions.l1("out", input, labels, null, LossFunctions.Reduction.MEAN_BY_COUNT, 1);
                    //L1 = sum abs error
                    expOut = Transforms.abs(inputArr.sub(labelsArr)).sum(1);
                    expOut = expOut.mean(Integer.MAX_VALUE);
                    break;
                case "l2":
                    lossInfo = LossFunctions.l2("out", input, labels, null, LossFunctions.Reduction.MEAN_BY_COUNT, 1);
                    //L2 = sum squared error
                    expOut = Transforms.pow(inputArr.sub(labelsArr),2.0).sum(1).mean(Integer.MAX_VALUE);
                    break;
                case "mcxent":
                    lossInfo = LossFunctions.mcxent("out", input, labels, null, LossFunctions.Reduction.MEAN_BY_COUNT, 1);
                    //mcxent = sum label * log(prob)
                    expOut = labelsArr.mul(Transforms.log(inputArr)).sum(1).mean(Integer.MAX_VALUE);
                    break;
                default:
                    throw new RuntimeException();
            }


            String msg = "test: " + lossInfo.getLossName();
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
    public void testDebugMSE(){

        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", new int[]{-1, nOut});
        SDVariable label = sd.var("labels", new int[]{-1, nOut});
        SDVariable weights = sd.one("weights", new int[]{1,1});

        INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
        INDArray labelsArr = Nd4j.randn(minibatch, nOut).muli(100);

        sd.associateArrayWithVariable(inputArr, predictions);
        sd.associateArrayWithVariable(labelsArr, label);

        SDVariable diff = predictions.sub(label);
        SDVariable preReduceLoss = sd.square(diff).mul(null, weights);


        SDVariable present = sd.neq(weights, 0.0);
        SDVariable presentBroadcast = sd.zerosLike("temp", label).add(present);
        SDVariable nonZeroWeights = sd.sum(presentBroadcast);

        SDVariable r = sd.mean(preReduceLoss, 1);
        SDVariable out = r.div("out", nonZeroWeights);

        INDArray outArr = sd.execAndEndResult();
    }

    @Test
    public void testDebugMSE2(){

        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", new int[]{-1, nOut});
        SDVariable label = sd.var("labels", new int[]{-1, nOut});
        SDVariable weights = sd.one("weights", new int[]{1,1});

        INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
        INDArray labelsArr = Nd4j.randn(minibatch, nOut).muli(100);

        sd.associateArrayWithVariable(inputArr, predictions);
        sd.associateArrayWithVariable(labelsArr, label);

        SDVariable diff = predictions.sub(label);
//        SDVariable preReduceLoss = sd.square(diff).mul(null, weights);
        SDVariable preReduceLoss = diff.mul(null, weights);

        SDVariable r = sd.mean(preReduceLoss, 1);
        SDVariable out = r.div("out", 4*10);

        INDArray outArr = sd.execAndEndResult();
    }

}
