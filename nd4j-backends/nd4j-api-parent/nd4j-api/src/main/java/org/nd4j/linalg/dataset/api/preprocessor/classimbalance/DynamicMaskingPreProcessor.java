package org.nd4j.linalg.dataset.api.preprocessor.classimbalance;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.string.NDArrayStrings;

import java.util.Arrays;

/**
 * Created by Alex on 09/06/2017.
 */
public class DynamicMaskingPreProcessor implements DataSetPreProcessor {

    public enum OutputType {Binary, Multiclass}

    private final OutputType outputType;
    private final INDArray maskingProbability;

    @java.beans.ConstructorProperties({"outputType", "maskingProbability"})
    public DynamicMaskingPreProcessor(@NonNull OutputType outputType, @NonNull INDArray maskingProbability) {
        this.outputType = outputType;
        this.maskingProbability = maskingProbability;

        if(outputType == OutputType.Binary && (maskingProbability.length() != 2 || !maskingProbability.isRowVector()) ){
            throw new IllegalArgumentException("Output type is set to Binary, but masking probability is not a row vector"
                    + " of length 2. Mask probabilities must have shape [1,2] with entries [P(maskClass0), P(maskClass1)]");
        } else if(outputType == OutputType.Multiclass && !maskingProbability.isRowVector() ){
            throw new IllegalArgumentException("Output type is set to Multiclass, but masking probability is not a row vector."
                    + " Mask probabilities must have shape [1,N] with entries [P(maskClass0), ..., P(maskClassN)] for N classes");
        }

        if(maskingProbability.minNumber().doubleValue() < 0.0 || maskingProbability.maxNumber().doubleValue() > 1.0){
            throw new IllegalArgumentException("All must values must be probabilities in range 0 to 1 inclusive. Got: "
                + Arrays.toString(maskingProbability.dup().data().asDouble()) );
        }
    }

    @Override
    public void preProcess(DataSet toPreProcess) {

        if(outputType == OutputType.Binary){
            INDArray labels = toPreProcess.getLabels();
            if(labels.size(1) != 1){
                throw new IllegalStateException("Output type is set to Binary, but labels.size(1) is " + labels.size(1)
                        + ". Use OutputType.Multiclass for multiple independent output classes");
            }

            int labelRank = labels.rank();
            INDArray labelMask = toPreProcess.getLabelsMaskArray();

            //Need to decide - based on the class - whether to mask it or not
            double mask0Prob = maskingProbability.getDouble(0);
            double mask1Prob = maskingProbability.getDouble(1);

            INDArray isClass1 = labels.dup();
            INDArray isClass0 = Transforms.not(labels);

            INDArray maskClass0 = Nd4j.createUninitialized(isClass0.shape());
            INDArray maskClass1 = Nd4j.createUninitialized(isClass1.shape());
            Nd4j.getExecutioner().exec(new BernoulliDistribution(maskClass0, mask0Prob));
            Nd4j.getExecutioner().exec(new BernoulliDistribution(maskClass1, mask1Prob));

            INDArray finalMask = maskClass0.muli(isClass0).addi(maskClass1.muli(isClass1));

            if(labelRank == 3){
                //Reshape from [minibatch,1,tsLength] to [minibatch,tsLength]
                finalMask.reshape(finalMask.ordering(), finalMask.size(0), finalMask.size(2));
            }

            //If we already have a mask: apply this on top. Otherwise: set it as the new mask
            if(labelMask != null){
                labelMask.muli(finalMask);  //i.e., an AND operation
            } else {
                toPreProcess.setLabelsMaskArray(finalMask);
            }
        } else {

            INDArray labels = toPreProcess.getLabels();
            if(labels.size(1) != maskingProbability.length()){
                throw new IllegalStateException("Output type is set to Multiclass; mismatch between number of mask " +
                        "probabilities and labels size (labels size: " + labels.size(1) + ", numProbs = " + maskingProbability.length());
            }

            INDArray maskPerClass = Nd4j.createUninitialized(labels.shape());
            int nClasses = maskingProbability.length();
            for( int i=0; i<nClasses; i++ ){
                double classIMaskProb = maskingProbability.getDouble(i);
                Nd4j.getExecutioner().exec(new BernoulliDistribution(maskPerClass.getColumn(i), classIMaskProb));
            }

            maskPerClass.muli(labels);

            INDArray maskOutput = maskPerClass.sum(1);

            if(toPreProcess.getLabelsMaskArray() == null){
                //No existing mask
                toPreProcess.setLabelsMaskArray(maskOutput);
            } else {
                //Apply the new mask on top of the existing mask
                toPreProcess.getLabelsMaskArray().muli(maskOutput);
            }
        }
    }
}
