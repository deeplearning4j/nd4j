package org.nd4j.autodiff.gradcheck;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.ACosh;
import org.nd4j.linalg.api.ops.impl.transforms.OldSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.Tan;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

@Slf4j
public class GradCheckTransforms {

    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testTransforms() {
        //Test reductions: final and only function
        Nd4j.getRandom().setSeed(12345);

        List<String> allFailed = new ArrayList<>();
        for (int i = 0; i < 27; i++) {

//            if(i == 6 || i == 15 || i == 17 || i == 18){
//                System.out.println("************** SKIPPING " + i + "****************");
//                continue;
//            }

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;
            SDVariable in = sd.var("in", new int[]{-1, nOut});

            INDArray ia = Nd4j.randn(minibatch, nOut);

            SDVariable t;
            INDArray expOut;
            switch (i) {
                case 0:
                    t = in.add(5.0);
                    expOut = ia.add(5.0);
                    break;
                case 1:
                    t = in.sub(5.0);
                    expOut = ia.sub(5.0);
                    break;
                case 2:
                    t = in.mul(2.5);
                    expOut = ia.mul(2.5);
                    break;
                case 3:
                    t = in.div(4.0);
                    expOut = ia.div(4);
                    break;
                case 4:
                    t = in.rsub(5.0);
                    expOut = ia.rsub(5.0);
                    break;
                case 5:
                    t = in.rdiv(1.0);
                    expOut = ia.rdiv(1.0);
                    break;
                case 6:
                    t = sd.pow(in, 2.5);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Transforms.pow(ia, 2.5, true);
                    break;
                case 7:
                    t = sd.sigmoid(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(2).subi(1.0);
                    expOut = Transforms.sigmoid(ia, true);
                    break;
                case 8:
                    t = sd.tanh(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(2).subi(1.0);
                    expOut = Transforms.tanh(ia, true);
                    break;
                case 9:
                    t = sd.tan(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Nd4j.getExecutioner().execAndReturn(new Tan(ia.dup()));
                    break;
                case 10:
                    t = sd.cos(in);
                    expOut = Transforms.cos(ia, true);
                    break;
                case 11:
                    t = sd.sin(in);
                    expOut = Transforms.sin(ia, true);
                    break;
                case 12:
                    t = sd.softplus(in);
                    expOut = Transforms.softPlus(ia, true);
                    break;
                case 13:
                    t = sd.log(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Transforms.log(ia, true);
                    break;
                case 14:
                    t = sd.neg(in);
                    expOut = ia.neg();
                    break;
                case 15:
                    t = sd.acos(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(1.8).subi(0.9);
                    expOut = Transforms.acos(ia, true);
                    break;
                case 16:
                    t = sd.acosh(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Nd4j.getExecutioner().execAndReturn(new ACosh(ia.dup()));
                    break;
                case 17:
                    t = sd.asin(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(1.8).subi(0.9);
                    expOut = Transforms.asin(ia, true);
                    break;
                case 18:
                    t = sd.atan(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(4).subi(2);
                    expOut = Transforms.atan(ia, true);
                    break;
                case 19:
                    t = sd.atanh(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(1.8).subi(0.9);
                    expOut = Transforms.atanh(ia, true);
                    break;
                case 20:
                    t = sd.cosh(in);
                    expOut = Transforms.cosh(ia, true);
                    break;
                case 21:
                    t = sd.cube(in);
                    expOut = Transforms.pow(ia, 3.0, true);
                    break;
                case 22:
                    t = sd.elu(in);
                    expOut = Transforms.elu(ia, true);
                    break;
                case 23:
                    t = sd.softmax(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Nd4j.getExecutioner().execAndReturn(new OldSoftMax(ia.dup()));
                    break;
                case 24:
                    t = sd.sqrt(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Transforms.sqrt(ia, true);
                    break;
                case 25:
                    t = sd.square(in);
                    expOut = Transforms.pow(ia, 2.0, true);
                    break;
                case 26:
                    t = sd.transpose(in);
                    expOut = ia.transpose().dup();
                    break;
                default:
                    throw new RuntimeException();
            }


            DifferentialFunction[] funcs = sd.functions();
            String name = funcs[0].opName();


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            SDVariable loss = sd.mean("loss", t);

            sd.associateArrayWithVariable(ia, in);
            sd.exec();
            INDArray out = t.getArr();

            assertEquals(msg, expOut, out);

            boolean ok = GradCheckUtil.checkGradients(sd);

//            assertTrue(msg, ok);
            if(!ok){
                allFailed.add(msg);
            }
        }

        if(allFailed.size() > 0){
            log.error("All failed transforms: " + allFailed);
            fail(allFailed.size() + " transforms failed");
        }
    }

    @Test
    public void powDeriv(){
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.rand(3,4));
        SDVariable pow2 = sd.pow("pow", in, 2.0);
        SDVariable sum = sd.sum(pow2);
        INDArray out = sd.execAndEndResult();

        INDArray expPow2 = Transforms.pow(in.getArr(),2.0);
        assertEquals(expPow2, pow2.getArr());

        sd.execBackwards();

        System.out.println(sd.grad("pow").getArr());

        //If L = sum(x^2) then dL/dx^i = 2 * x_i
        INDArray expGrad = in.getArr().mul(2);
        assertEquals(expGrad, sd.grad("pow").getArr());
    }

    @Test
    public void testSigmoidDerivative(){
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.rand(3,4));
        SDVariable sigmoid = sd.sigmoid("sigmoid", in);
        SDVariable sum = sd.sum(sigmoid);
        INDArray out = sd.execAndEndResult();

        INDArray expS = Transforms.sigmoid(in.getArr());
        assertEquals(expS, sigmoid.getArr());

        //If L = sum(sigmoid(x)) then dL/dx_i = s(x) * (1-s(x))
        INDArray expDeriv = expS.mul(expS.rsub(1.0));

        sd.execBackwards();

        System.out.println(sd.grad("sigmoid").getArr());

        assertEquals(expDeriv, sd.grad("sigmoid").getArr());
    }

    @Test
    public void testTanhDerivative(){
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.rand(3,4)).muli(2).subi(1.0);
        SDVariable tanh = sd.tanh("tanh", in);
        SDVariable sum = sd.sum(tanh);
        INDArray out = sd.execAndEndResult();

        INDArray exp = Transforms.tanh(in.getArr());
        assertEquals(exp, tanh.getArr());

        //If L = sum(tanh(x)) then dL/dx_i = 1 - (tanh(x))^2
        INDArray expDeriv = exp.mul(exp).rsub(1.0);

        sd.execBackwards();

        System.out.println(sd.grad("tanh").getArr());

        assertEquals(expDeriv, sd.grad("tanh").getArr());
    }
}
