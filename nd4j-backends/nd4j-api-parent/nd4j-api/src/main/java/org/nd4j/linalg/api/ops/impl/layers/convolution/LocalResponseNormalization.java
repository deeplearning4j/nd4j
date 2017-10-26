package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * LocalResponseNormalization operation
 */
@Slf4j
@Getter
public class LocalResponseNormalization extends DynamicCustomOp {



    protected LocalResponseNormalizationConfig config;


    @Builder(builderMethodName = "builder")
    public LocalResponseNormalization(SameDiff sameDiff, DifferentialFunction[] inputFunctions,INDArray[] inputs, INDArray[] outputs,boolean inPlace,LocalResponseNormalizationConfig config) {
        super(null,sameDiff, inputFunctions, inPlace);
        this.config = config;
        if(inputs != null) {
            getInputArguments().addAll(Arrays.asList(inputs));
        }

        if(outputs!= null) {
            getOutputArguments().addAll(Arrays.asList(outputs));
        }

        addArgs();
    }


    public LocalResponseNormalization() {}


    private void addArgs() {
        getTArguments().add(config.getAlpha());
        getTArguments().add(config.getBeta());
        getTArguments().add(config.getBias());
        getTArguments().add(config.getDepth());
    }

    @Override
    public String opName() {
        return "lrn";
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        LocalResponseNormalizationDerivative localResponseNormalizationDerivative = LocalResponseNormalizationDerivative.derivativeBuilder()
                .inPlace(inPlace)
                .sameDiff(sameDiff)
                .inputFunctions(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .config(config)
                .build();
        ret.addAll(Arrays.asList(localResponseNormalizationDerivative.getOutputFunctions()));

        return ret;
    }

}
