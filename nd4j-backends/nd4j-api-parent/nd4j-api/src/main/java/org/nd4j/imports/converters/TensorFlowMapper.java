package org.nd4j.imports.converters;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.ops.DefaultOpConverter;
import org.tensorflow.framework.NodeDef;

import java.util.Set;

@Slf4j
public class TensorFlowMapper implements NodeMapper<NodeDef> {
    private static final TensorFlowMapper INSTANCE = new TensorFlowMapper();

    protected TensorFlowMapper() {

    }

    public static TensorFlowMapper getInstance() {
        return INSTANCE;
    }

    @Override
    public TOp asIntermediate(NodeDef node, TGraph graph) {
        // first we try to use special converters
        DifferentialFunction converter = DifferentialFunctionClassHolder.getInstance().getInstance(node.getOp().toLowerCase());
        if(converter == null)
            converter = DifferentialFunctionClassHolder.getInstance().getInstance(DefaultOpConverter.getInstance().opName());
        return converter.asIntermediateRepresentation(node, graph);

    }

    public Set<String> knownOps() {
        return DifferentialFunctionClassHolder.getInstance().opNames();
    }
}
