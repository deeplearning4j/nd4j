package org.nd4j.imports.converters;

import lombok.NonNull;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;

public interface NodeMapper<T> {

    TOp asIntermediate(@NonNull T node, @NonNull TGraph graph);
}
