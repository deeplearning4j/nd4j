package org.nd4j.autodiff.gradcheck;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

@Slf4j
public class GradCheckTransforms {

    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testTransformsSimple() {
        //Test reductions: final and only function
        Nd4j.getRandom().setSeed(12345);

        for (int i = 0; i < 27; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;
            SDVariable in = sd.var("in", new int[]{-1, nOut});

            INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(10);

            SDVariable t;
            switch (i) {
                case 0:
                    t = in.add(5.0);
                    break;
                case 1:
                    t = in.sub(5.0);
                    break;
                case 2:
                    t = in.mul(2.5);
                    break;
                case 3:
                    t = in.div(4.0);
                    break;
                case 4:
                    t = in.rsub(5.0);
                    break;
                case 5:
                    t = in.rdiv(1.0);
                    break;
                case 6:
                    t = sd.pow(in, 2.5);
                    break;
                case 7:
                    t = sd.sigmoid(in);
                    break;
                case 8:
                    t = sd.tanh(in);
                    break;
                case 9:
                    t = sd.tan(in);
                    break;
                case 10:
                    t = sd.cos(in);
                    break;
                case 11:
                    t = sd.sin(in);
                    break;
                case 12:
                    t = sd.softplus(in);
                    break;
                case 13:
                    t = sd.log(in);
                    inputArr = Nd4j.rand(minibatch, nOut);
                    break;
                case 14:
                    t = sd.neg(in);
                    break;
                case 15:
                    t = sd.acos(in);
                    break;
                case 16:
                    t = sd.acosh(in);
                    break;
                case 17:
                    t = sd.asin(in);
                    break;
                case 18:
                    t = sd.atan(in);
                    break;
                case 19:
                    t = sd.atanh(in);
                    break;
                case 20:
                    t = sd.cosh(in);
                    break;
                case 21:
                    t = sd.cube(in);
                    break;
                case 22:
                    t = sd.elu(in);
                    break;
                case 23:
                    t = sd.softmax(in);
                    break;
                case 24:
                    t = sd.sqrt(in);
                    inputArr = Nd4j.rand(minibatch, nOut);
                    break;
                case 25:
                    t = sd.square(in);
                    break;
                case 26:
                    t = sd.transpose(in);
                    break;
                default:
                    throw new RuntimeException();
            }


            String msg = "test: " + i;
            log.info("*** Starting test: " + msg);

            SDVariable loss = sd.mean("loss", t);


            sd.associateArrayWithVariable(inputArr, in);

            boolean ok = GradCheckUtil.checkGradients(sd);

            assertTrue(msg, ok);
        }
    }
}
