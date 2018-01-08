package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Map;


/**
 * An alias for tensorflow's atrous
 * convolution (vs being built in to pytorch)
 * operation
 */
@Slf4j
@Getter
public class AtrousConv2D extends Conv2D {


    @Builder(builderMethodName = "atrousBuilder")
    public AtrousConv2D(SameDiff sameDiff,
                        SDVariable[] inputFunctions,
                        INDArray[] inputArrays, INDArray[] outputs,
                        Conv2DConfig conv2DConfig) {
        super(sameDiff, inputFunctions, inputArrays, outputs, conv2DConfig);
        this.sameDiff = sameDiff;

        this.conv2DConfig = conv2DConfig;
        addArgs();
    }

    public AtrousConv2D() {
    }

    protected void addArgs() {
        addIArgument(conv2DConfig.getKh());
        addIArgument(conv2DConfig.getKw());
        addIArgument(conv2DConfig.getSy());
        addIArgument(conv2DConfig.getSx());
        addIArgument(conv2DConfig.getPh());
        addIArgument(conv2DConfig.getPw());
        addIArgument(conv2DConfig.getDh());
        addIArgument(conv2DConfig.getDw());
        addIArgument(ArrayUtil.fromBoolean(conv2DConfig.isSameMode()));

    }



    @Override
    public Map<String, Object> propertiesForFunction() {
        return conv2DConfig.toProperties();
    }

    @Override
    public String opName() {
        return "atrous_conv2d";
    }


    @Override
    public String onnxName() {
        return "Conv";
    }

    @Override
    public String tensorflowName() {
        return "Dilation2D";
    }
}
