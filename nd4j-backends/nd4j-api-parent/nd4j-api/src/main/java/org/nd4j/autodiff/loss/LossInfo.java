package org.nd4j.autodiff.loss;

import lombok.Builder;
import lombok.Getter;
import org.nd4j.autodiff.samediff.SDVariable;

@Builder(builderClassName = "Builder")
@Getter
public class LossInfo {
    private String lossName;
    private LossFunctions.Reduction reduction;
    private SDVariable loss;
    private SDVariable label;
    private SDVariable predictions;
    private SDVariable weights;

}
