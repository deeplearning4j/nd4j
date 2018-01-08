package org.nd4j.autodiff.loss;

import com.google.common.base.Preconditions;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class LossFunctions {

    private static final int[] SCALAR = new int[]{1,1};

    public enum Reduction {
        /**
         * No reduction. Output is the same shape as the predictions/labels.
         * Weights (if any) are applied. Dimension args are ignored.
         * Example: 2d input, MSE.
         * Output: sqDiff(predictions,labels) -> shape same as input/labels
         */
        NONE,
        /**
         * Reduce as normal along the specified dimensions, but don't sum/mean etc the remaining
         * dimensions.
         * Example: 2d input, MSE loss along dimension 1.
         * Output: mean(weights * sqDiff(predictions,labels),1) -> shape [dim0,1]
         */
        SPECIFIED_DIMS,
        /**
         * Sum across the remaining dimensions, returning a scalar
         * Example: 2d input, MSE loss along dimension 1.
         * Output: mse_per_ex = mean(weights * sqDiff(predictions,labels),1)          *Same as SPECIFIED_DIMS*
         *         output = sum(mse_per_ex)
         */
        SUM,
        /**
         * Weighted mean: sum(weights * loss) / sum(weights)
         * Example: 2d input, MSE loss along dimension 1.
         * Output: mse_per_ex = mean(weights * sqDiff(predictions,labels),1)          *Same as SPECIFIED_DIMS*
         *         output = sum(mse_per_ex) / sum(weights)
         *
         * NOTE: if weights array is not provided, then weights default to (effectively) 1.0 for all entries - and hence
         * MEAN_BY_WEIGHT is equivalent to SUM (as sum(1.0) = 1.0)
         */
        MEAN_BY_WEIGHT,

        /**
         * Weighted mean: sum(weights * loss) / count(weights != 0)
         * Example: 2d input, MSE loss along dimension 1.
         * Output: mse_per_ex = mean(weights * sqDiff(predictions,labels),1)          *Same as SPECIFIED_DIMS*
         *         output = sum(mse_per_ex) / count(weights != 0)
         *
         * NOTE: if weights array is not provided, then weights default to scalar 1.0 and hence MEAN_BY_COUNT
         * is equivalent to
         */
        MEAN_BY_COUNT

    }

    private LossFunctions(){ }




    public Object lossMSE(String outputName, SDVariable predictions, SDVariable label, SDVariable weights,
                          Reduction reduction, int... dimensions){
        Preconditions.checkNotNull(predictions, "Predictions variable cannot be null");
        Preconditions.checkNotNull(label, "Label variable cannot be null");
        Preconditions.checkNotNull(reduction, "Reduction enumeration cannot be null");
        SameDiff sd = predictions.getSameDiff();

        if(weights == null){
            weights = sd.one("weights", SCALAR);
        }



        SDVariable diff = predictions.sub(label);
        String name = (reduction == Reduction.NONE ? outputName : null);
        SDVariable preReduceLoss = sd.square(diff).mul(name, weights);

        LossInfo.Builder b = LossInfo.builder()
                .reduction(reduction)
                .label(label)
                .predictions(predictions);


        switch (reduction){
            case NONE:
                //Return same shape as predictions/labels
                b.loss(preReduceLoss);
                break;
            case SPECIFIED_DIMS:
                //Reduce along specified dimensions
                b.loss(sd.mean(outputName, preReduceLoss, dimensions));
            case SUM:
                SDVariable m = sd.mean(preReduceLoss, dimensions);
                b.loss(sd.sum(outputName, m));
                break;
            case MEAN_BY_WEIGHT:
                //reduce along dims + reduce along remaining dims == reduce along *all* dims
                SDVariable m2 = sd.mean(preReduceLoss);
                SDVariable weightSum = sd.sum(weights);
                b.loss(m2.div(outputName, weightSum));
                break;
            case MEAN_BY_COUNT:
                SDVariable m3 = sd.mean(preReduceLoss);
                SDVariable nonZeroWeights = weights.
                break;
        }



    }


    private static SDVariable nonZeroCount(SDVariable weights, SDVariable labels){
        SameDiff sd = weights.getSameDiff();

        SDVariable present = sd.neq(weights, 0.0);


    }

}
