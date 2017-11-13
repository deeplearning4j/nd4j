package org.nd4j.imports.converters;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.ops.DefaultOpConverter;
import onnx.OnnxProto3.NodeProto;

import java.util.Set;

@Slf4j
public class OnnxMapper implements NodeMapper<NodeProto> {
    private static final OnnxMapper INSTANCE = new OnnxMapper();

    protected OnnxMapper() {

    }

    public static OnnxMapper getInstance() {
        return INSTANCE;
    }

    @Override
    public TOp asIntermediate(NodeProto node, TGraph graph) {
        // first we try to use special converters
        DifferentialFunction converter = DifferentialFunctionClassHolder.getInstance().getInstance(node.getName().toLowerCase());
        if(converter == null)
            converter = DifferentialFunctionClassHolder.getInstance().getInstance(DefaultOpConverter.getInstance().opName());
        return converter.asIntermediateRepresentation(node, graph);

    }

    public Set<String> knownOps() {
        return DifferentialFunctionClassHolder.getInstance().opNames();
    }
}
