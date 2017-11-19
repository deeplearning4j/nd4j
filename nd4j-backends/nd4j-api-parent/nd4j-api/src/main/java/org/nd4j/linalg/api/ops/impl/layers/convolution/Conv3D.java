package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


/**
 * Conv3D operation
 */
@Slf4j
@Getter
public class Conv3D extends DynamicCustomOp {

    protected Conv3DConfig config;

    public Conv3D() {}

    @Builder(builderMethodName = "builder")
    public Conv3D(SameDiff sameDiff, DifferentialFunction[] inputFunctions,INDArray[] inputs, INDArray[] outputs,Conv3DConfig conv3DConfig) {
        super(null,sameDiff, inputFunctions, false);
        setSameDiff(sameDiff);
        setArgs(inputFunctions);
        if(inputs != null)
            getInputArguments().addAll(Arrays.asList(inputs));
        if(outputs != null)
            getOutputArguments().addAll(Arrays.asList(outputs));
        this.config = conv3DConfig;
        addArgs();

    }


    private void addArgs() {
        getIArguments().add(getConfig().getDT());
        getIArguments().add(getConfig().getDW());
        getIArguments().add(getConfig().getDH());
        getIArguments().add(getConfig().getPT());
        getIArguments().add(getConfig().getPW());
        getIArguments().add(getConfig().getPH());
        getIArguments().add(getConfig().getDilationT());
        getIArguments().add(getConfig().getDilationW());
        getIArguments().add(getConfig().getDilationH());
        getIArguments().add(getConfig().getAT());
        getIArguments().add(getConfig().getAW());
        getIArguments().add(getConfig().getAH());
        getIArguments().add(fromBoolean(getConfig().isBiasUsed()));

    }

    @Override
    public String opName() {
        return "conv3d";
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Conv3DDerivative conv3DDerivative = Conv3DDerivative.derivativeBuilder()
               .conv3DConfig(config)
                .inputFunctions(args())
                .outputs(this.getOutputArguments().toArray(new INDArray[this.getOutputArguments().size()]))
                .inputFunctions(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .sameDiff(sameDiff)
                .build();
        ret.addAll(Arrays.asList(conv3DDerivative.getOutputFunctions()));
        return ret;
    }

    @Override
    public TOp asIntermediateRepresentation(NodeDef node, TGraph graph) {
        return super.asIntermediateRepresentation(node, graph);
    }

    @Override
    public TOp asIntermediateRepresentation(OnnxProto3.NodeProto node, TGraph graph, Map<String, OnnxProto3.AttributeProto> attributesForNode) {
        return super.asIntermediateRepresentation(node, graph, attributesForNode);
    }

    @Override
    public String onnxName() {
        return "Conv";
    }

    @Override
    public String tensorflowName() {
        return "conv3d";
    }
}
