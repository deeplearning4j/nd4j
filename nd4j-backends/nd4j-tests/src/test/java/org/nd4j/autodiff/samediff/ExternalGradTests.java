package org.nd4j.autodiff.samediff;

import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class ExternalGradTests {

    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }


    @Test
    public void testExternalGrad(){

        Activation[] afns = new Activation[]{
                Activation.TANH,
                Activation.SIGMOID,
//                Activation.ELU,           //Fails
//                Activation.IDENTITY,
//                Activation.SOFTPLUS,
//                Activation.SOFTSIGN,
        };

        for(Activation a : afns) {
            System.out.println("Test:: " + a);
            Nd4j.getRandom().setSeed(12345);
            INDArray inArr = Nd4j.rand(1, 4);

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", inArr.dup());
//            SDVariable s = sd.tanh("s", in);
            SDVariable s = a.asSameDiff("s", sd, in);

            INDArray out = sd.execAndEndResult();
            INDArray outEx = Nd4j.getExecutioner().execAndReturn(a.asTransform(inArr, true));

            assertEquals(outEx, out);

            INDArray externalGrad = Nd4j.rand(inArr.shape());
            Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> p = sd.execBackwardsTEST(externalGrad);

            Map<SDVariable, DifferentialFunction> m = p.getFirst();
            List<DifferentialFunction> l = p.getSecond();

            SameDiff gradFn = sd.getFunction("grad-external");
//            for (SDVariable sdv : gradFn.variables()) {
//                System.out.println(sdv.getVarName() + "\t" + sdv.getArr());
//            }

            INDArray expGradIn = Nd4j.getExecutioner().execAndReturn(a.asTransformDerivative(inArr,true));
            expGradIn.muli(externalGrad);

            SDVariable inGradVar = gradFn.getVariable("in-grad");
            INDArray actGradIn = inGradVar.getArr();
            System.out.println("Exp grad in:\n" + expGradIn);
            System.out.println("Exp grad in:\n" + Arrays.toString(expGradIn.dup().data().asFloat()));
            System.out.println("Act grad in:\n" + actGradIn);
            System.out.println("Act grad in:\n" + Arrays.toString(actGradIn.dup().data().asFloat()));
            assertEquals(expGradIn, actGradIn);
            System.out.println("-----------------------------");
        }
    }
}
