package org.nd4j.imports.converters.tf;

import lombok.NonNull;
import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.nd4j.imports.converters.ExternalNode;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.NodeDef;

public abstract class BaseTensorFlowNode implements ExternalNode<NodeDef> {
    protected NodeDef tfNode;
    protected TGraph tGraph;

    protected BaseTensorFlowNode() {
        //
    }

    protected BaseTensorFlowNode(@NonNull Object nodeDef, @NonNull TGraph tGraph) {
        this.tfNode = (NodeDef) nodeDef;
        this.tGraph = tGraph;
    }

    /**
     * This method returns given TF node as TNode
     *
     * @return
     */
    @Override
    public abstract TNode asIntermediateRepresentation(@NonNull NodeDef node, @NonNull TGraph graph);

    /**
     * This method returns given TF node as ND4j Op
     *
     * @return
     */
    @Override
    public Op asExecutableOperation(@NonNull NodeDef node, @NonNull TGraph graph) {
        return null;
    }

    protected TNode buildBasicNode(@NonNull NodeDef node, @NonNull TGraph graph) {
        val tNode = TNode.builder()
                .name(node.getName())
                .build();

        return tNode;
    }
}
