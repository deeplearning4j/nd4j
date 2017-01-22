package org.nd4j.linalg.lossfunctions;

import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.After;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.lossfunctions.impl.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by Alex on 08/08/2016.
 */
@Slf4j
public class LossFunctionGradientChecks extends BaseNd4jTest {

    private static final double maxRelError = 5.0; //5% relative error
    DataBuffer.Type initialType;

    public LossFunctionGradientChecks(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Before
    public void before() throws Exception {
        super.before();

        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        Nd4j.getRandom().setSeed(123);
    }

    @Test
    public void testLossFunctionGradients(){

        //Case Default: loss functions with labels taking real values
        ILossFunction[] lossFns = {new LossMSE(),
        new LossL1(),
        new LossMAE(),
        new LossL2(),
        new LossMAPE(),
        new LossMSLE(),
        new LossCosineProximity(),
        new LossPoisson()
        };

        String[] activationFns = new String[]{"cube",
                "leakyrelu",
                "identity",
                "tanh",
                "hardtanh",
                "softsign",
                "hardsigmoid",
                "sigmoid",
                "elu",
                "softplus",
                "relu"};

        System.out.println ("================ Running loss functions, regression =======================");
        for (int i=0; i< lossFns.length; i++) {
            System.out.print("===");
            System.out.println(lossFns[i].toString());
            doGradientCheck(lossFns[i], activationFns, "default");
        }

        //Case B: loss functions with labels taking binary values (one hot or not)
        //Binary cross entropy - labels are one hot, used only with activation softmax
        //MCXENT, NLL should work without this assumption

        activationFns = new String [] {
                "softmax", //only run with one hot
                "hardsigmoid",
                "sigmoid"};

        lossFns = new ILossFunction[]{
                new LossBinaryXENT(), //only run with one hot
                new LossMCXENT(),
                new LossNegativeLogLikelihood()
        };

        System.out.println ("================ Running loss functions, classification =======================");
        for (int i=0; i< lossFns.length; i++) {
            System.out.println(lossFns[i].toString());
            System.out.println ("===One hot");
            doGradientCheck(lossFns[i],activationFns,"onehot");
            System.out.println ("===Not one hot");
            doGradientCheck(lossFns[i],activationFns,"prob");
        }

        //Case C: loss functions with labels taking -1 or 1, same as above shifted range
        //Hinge loss and squared hinge loss, written for label values -1 or 1
        lossFns = new ILossFunction[]{
                new LossHinge(),
                new LossSquaredHinge()
        };
        activationFns = new String [] {
                "softsign",
                "hardtanh",
                "tanh"};

        System.out.println ("================ Running loss functions, classification with hinge loss =======================");
        for (int i=0; i< lossFns.length; i++) {
            System.out.println(lossFns[i].toString());
            System.out.println ("===One hot");
            doGradientCheck(lossFns[i],activationFns,"onehot");
            System.out.println ("===Not one hot");
            doGradientCheck(lossFns[i],activationFns,"prob");
        }

    }


    public static void doGradientCheck(ILossFunction lossfn, String [] activationFns, String labelType) {
        double epsilon = 1e-6;
        int totalNFailures = 0;
        double maxRelError = 5.0; // in %

        int[] labelSizes = new int[]{1, 2, 10};

        if (labelType.equals("prob") && lossfn instanceof LossBinaryXENT) return;
        if (labelType.equals("onehot") && lossfn instanceof LossBinaryXENT) labelSizes = new int[] {2};

        for (int i = 0; i < activationFns.length; i++) {

            System.out.println("Running checks for "+activationFns[i]);

            String activationS = activationFns[i];
            IActivation activation = Activation.fromString(activationS).getActivationFunction();

            List<INDArray> labelList = makeLabels(labelType,labelSizes);
            List<INDArray> preOutputList = makeLabels("default",labelSizes);


            for (int j=0; j<labelSizes.length; j++) {

                if(labelType.equals("prob") && activationS.equals("softmax")) continue;
                if(labelSizes[j]==1 && activationS.equals("softmax")) continue;
                if(labelSizes[j]==1 && lossfn instanceof LossCosineProximity) continue;

                System.out.println("\tRunning check for length " + labelSizes[j]);

                INDArray label = labelList.get(j);
                if (lossfn instanceof LossHinge || lossfn instanceof LossSquaredHinge) {label.muli(2).subi(1);}

                INDArray preOut = preOutputList.get(j);
                INDArray grad = lossfn.computeGradient(label.dup(),preOut.dup(),activation,null);

                NdIndexIterator iterPreOut = new NdIndexIterator(preOut.shape());

                while (iterPreOut.hasNext()) {
                    int[] next = iterPreOut.next();
                    //checking gradient with total score wrt to each output feature in label
                    double before = preOut.getDouble(next);
                    preOut.putScalar(next, before + epsilon);
                    double scorePlus = lossfn.computeScore(label.dup(), preOut.dup(), activation, null, true);
                    preOut.putScalar(next, before - epsilon);
                    double scoreMinus = lossfn.computeScore(label.dup(), preOut.dup(), activation, null, true);
                    preOut.putScalar(next, before);

                    double scoreDelta = scorePlus - scoreMinus;
                    double numericalGradient = scoreDelta / (2 * epsilon);
                    double analyticGradient = grad.getDouble(next);
                    double relError = Math.abs(analyticGradient - numericalGradient) * 100 / (Math.abs(numericalGradient));
                    if( analyticGradient == 0.0 && numericalGradient == 0.0 ) relError = 0.0;
                    if (relError > maxRelError || Double.isNaN(relError)) {
                        System.out.println("\t\tParam " + Arrays.toString(next) + " FAILED: grad= " + analyticGradient + ", numericalGrad= " + numericalGradient
                               + ", relErrorPerc= " + relError + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                        totalNFailures++;
                    } else {
                        //System.out.println("\t\tParam " + Arrays.toString(next) + " passed: grad= " + analyticGradient + ", numericalGrad= " + numericalGradient
                        //        + ", relError= " + relError + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                    }
                }
                System.out.println("\t\tlabels:"+label.toString());
                System.out.println("\t\tpreout:"+preOut.toString());
            }
        }

        if(totalNFailures > 0) System.out.println("DONE:\n\tGradient check failed for loss function:"+lossfn.toString()+"; total num failures = " + totalNFailures);
        assert(totalNFailures == 0);
    }


    /*
        Return valid labels for different activation functions with given labelSize
     */
    public static List<INDArray> makeLabels(String labelType,int[]labelSize) {
        List<INDArray> returnVals = new ArrayList<>(labelSize.length);
        for (int i=0; i< labelSize.length; i++) {
            int aLabelSize = labelSize[i];
            Random r = new Random();
            double[] someVals = new double[aLabelSize];
            boolean oneHotSet = false;
            double transformVal;
            for (int j=0; j<aLabelSize; j++) {
                switch (labelType) {
                    case "prob":
                        transformVal = r.nextBoolean()? 1 : 0;
                        if (aLabelSize == 1) transformVal = 0;
                        break;
                    case "onehot":
                        //one hot value
                        if (!oneHotSet) {
                            oneHotSet = r.nextBoolean();
                            transformVal = oneHotSet ? 1 : 0;
                        }
                        else {
                            transformVal = 0;
                        }
                        break;
                    default:
                        transformVal = 0.5*r.nextDouble()+0.4;
                        break;
                }
                someVals[j] = transformVal;
            }
            if (!oneHotSet && (labelType.equals("onehot")|| labelType.equals("onehotS"))) {
                //pick an index to make 1
                transformVal = r.nextDouble();
                transformVal *= aLabelSize;
                someVals[(int)Math.floor(transformVal)] = 1;
            }
            returnVals.add(Nd4j.create(someVals));
        }
        return returnVals;
    }

    @After
    public void after() {
        DataTypeUtil.setDTypeForContext(this.initialType);
        System.out.println("AFTER DATATYPE HERE: "+ Nd4j.dataType());
    }

    @Override
    public char ordering() {
        return 'f';
    }

}
