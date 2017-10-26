package org.nd4j.imports.converters.tf;

import lombok.NonNull;
import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

public class Sum extends BaseTensorFlowNode{

    public Sum() {
        super();
    }

    public Sum(@NonNull NodeDef nodeDef, @NonNull TGraph graph) {
        super(nodeDef, graph);
    }

    @Override
    public String opName() {
        return "Sum";
    }

    /**
     * This method returns given TF node as TNode
     *
     * @return
     */
    @Override
    public TNode asIntermediateRepresentation(@NonNull NodeDef node, @NonNull TGraph graph) {
        val tNode = TNode.builder()
                .opName(opName())
                .opNum(0)
                .build();



        return tNode;
    }
}
