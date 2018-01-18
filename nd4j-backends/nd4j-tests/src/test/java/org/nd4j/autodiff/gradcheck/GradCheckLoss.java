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
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@Slf4j
public class GradCheckLoss {

    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testLossSimple2d() {
        Nd4j.getRandom().setSeed(12345);

        for (String fn : new String[]{"mse", "l1", "l2", "mcxent"}) {

            for (LossFunctions.Reduction reduction : new LossFunctions.Reduction[]{
                    LossFunctions.Reduction.MEAN_BY_COUNT, LossFunctions.Reduction.MEAN_BY_WEIGHT, LossFunctions.Reduction.SUM}) {

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
                        lossInfo = LossFunctions.mse("out", input, labels, null, reduction, 1);
                        expOut = inputArr.sub(labelsArr);
                        expOut.muli(expOut);
                        expOut = expOut.mean(Integer.MAX_VALUE);
                        break;
                    case "l1":
                        lossInfo = LossFunctions.l1("out", input, labels, null, reduction, 1);
                        //L1 = sum abs error
                        expOut = Transforms.abs(inputArr.sub(labelsArr)).sum(1);
                        expOut = expOut.mean(Integer.MAX_VALUE);
                        break;
                    case "l2":
                        lossInfo = LossFunctions.l2("out", input, labels, null, reduction, 1);
                        //L2 = sum squared error
                        expOut = Transforms.pow(inputArr.sub(labelsArr), 2.0).sum(1).mean(Integer.MAX_VALUE);
                        break;
                    case "mcxent":
                        lossInfo = LossFunctions.mcxent("out", input, labels, null, reduction, 1);
                        //mcxent = sum label * log(prob)
                        expOut = labelsArr.mul(Transforms.log(inputArr)).sum(1).mean(Integer.MAX_VALUE);
                        break;
                    default:
                        throw new RuntimeException();
                }


                String msg = "test: " + lossInfo.getLossName() + ", reduction=" + reduction;
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
    }

    @Test
    public void testLossWeights2d() {

        String[] weightTypes = new String[]{"none", "per-example", "per-output", "per-example-output"};

        Nd4j.getRandom().setSeed(12345);

        int nOut = 4;
        int minibatch = 10;

        for (String weightType : weightTypes) {

            for (boolean binary : new boolean[]{true, false}) {  //Binary mask (like DL4J) or arbitrary weights?

                int[] weightShape;
                switch (weightType) {
                    case "none":
                        weightShape = null;
                        break;
                    case "per-example":
                        weightShape = new int[]{minibatch, 1};
                        break;
                    case "per-output":
                        weightShape = new int[]{1, nOut};
                        break;
                    case "per-example-output":
                        weightShape = new int[]{minibatch, nOut};
                        break;
                    default:
                        throw new RuntimeException("Unknown type: " + weightType);
                }

                INDArray weightArr = null;
                if (!"none".equals(weightType)) {
                    if (binary) {
                        weightArr = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(weightShape), 0.5));
                    } else {
                        weightArr = Nd4j.rand(weightShape).muli(2.0);
                    }
                }

                for (LossFunctions.Reduction reduction : new LossFunctions.Reduction[]{
                        LossFunctions.Reduction.MEAN_BY_COUNT, LossFunctions.Reduction.MEAN_BY_WEIGHT, LossFunctions.Reduction.SUM}) {

                    for (String fn : new String[]{"mse", "l1", "l2", "mcxent"}) {

                        SameDiff sd = SameDiff.create();


                        SDVariable input = sd.var("in", new int[]{-1, nOut});
                        SDVariable labels = sd.var("labels", new int[]{-1, nOut});
                        SDVariable weight = null;
                        if (!"none".equals(weightType)) {
                            weight = sd.var("weights", weightArr);
                        }

                        INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
                        INDArray labelsArr = Nd4j.randn(minibatch, nOut).muli(100);

                        LossInfo lossInfo;
                        switch (fn) {
                            case "mse":
                                lossInfo = LossFunctions.mse("out", input, labels, weight, reduction, 1);
                                break;
                            case "l1":
                                lossInfo = LossFunctions.l1("out", input, labels, weight, reduction, 1);
                                //L1 = sum abs error
                                break;
                            case "l2":
                                lossInfo = LossFunctions.l2("out", input, labels, weight, reduction, 1);
                                //L2 = sum squared error
                                break;
                            case "mcxent":
                                lossInfo = LossFunctions.mcxent("out", input, labels, weight, reduction, 1);
                                //mcxent = sum label * log(prob)
                                break;
                            default:
                                throw new RuntimeException();
                        }


                        String msg = "lossFn=" + fn + ", reduction=" + reduction + ", weightType=" + weightType + ", binaryWeight=" + binary;
                        log.info("*** Starting test: " + msg);

                        sd.associateArrayWithVariable(inputArr, input);
                        sd.associateArrayWithVariable(labelsArr, labels);
                        if (weight != null) {
                            sd.associateArrayWithVariable(weightArr, weight);
                        }

                        INDArray out = sd.execAndEndResult();
                        assertEquals(1, out.length());

                        boolean ok = GradCheckUtil.checkGradients(sd);

                        assertTrue(msg, ok);
                    }
                }
            }
        }
    }
}
