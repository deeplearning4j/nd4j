package org.nd4j.imports.graphmapper;

import lombok.Data;
import org.nd4j.autodiff.samediff.SameDiff;

@Data
public class ImportState<GRAPH_TYPE> {
    private int nodeCount;
    private SameDiff sameDiff;
    private GRAPH_TYPE graph;

    public void incrementNodeCount() {
        nodeCount++;
    }

}
