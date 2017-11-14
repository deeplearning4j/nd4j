package org.nd4j.imports.graphmapper;

import com.google.common.collect.Maps;
import com.google.common.primitives.Ints;
import lombok.val;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.samediff.SDGraph;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Base implementation for importing a graph
 * @param <GRAPH_TYPE> the type of graph
 * @param <NODE_TYPE> the type of node
 * @param <ATTR_TYPE> the attribute type
 */
public abstract class BaseGraphMapper<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE> implements GraphMapper<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE> {


    /**
     * This method converts given TF
     * @param tfGraph
     * @return
     */
    @Override
    public SameDiff importGraph(GRAPH_TYPE tfGraph) {
        SDGraph graph = new SDGraph(true);

        SameDiff diff = SameDiff.builder()
                .graph(graph)
                .variableMap(Maps.newHashMap())
                .build();

        graph.setSameDiff(diff);

        Map<String, Integer> reverseVertexMap = new HashMap<>();

        int nodesCnt = 0;
        val skipPoint = new AtomicLong(0);
        Set<String> skipList = new HashSet<>();
        val tfNodesList = getNodeList(tfGraph);
        for (NODE_TYPE tfNode : tfNodesList) {
            //log.debug("Node opName: {}; Op: {};", tfNode.getName(), tfNode.getOp());

            if (shouldSkip(tfNode)) {
                continue;
            }

            // if we've used forward-scan (i.e. for loops ) we can already have this node mapped
            if (alreadySeen(tfNode))
                continue;





            if (isVariableNode(tfNode)) {
                List<Integer> dimensions = new ArrayList<>();
                SDVariable variable = SDVariable.builder()
                        .sameDiff(diff)
                        .varName(getName(tfNode))
                        .build();

                SDVariable varInformation = SDVariable.builder()
                        .varName(getName(tfNode))
                        .build();

                NDArrayVertex vertex = new NDArrayVertex(diff,++nodesCnt,0, varInformation);

                reverseVertexMap.put(getName(tfNode), nodesCnt);

                int[] arrayShape = null;
                Map<String, ATTR_TYPE> attributes = getAttrMap(tfNode);


                if (attributes.containsKey(dTypeKey())) {
                    ATTR_TYPE dtype = attributes.get(dTypeKey());

                    // dtype.getList();
                }

                if (attributes.containsKey(shapeKey())) {
                    ATTR_TYPE shape = attributes.get(shapeKey());
                    int dims = getShapeFromAttr(shape).length;
                    if (dims > 0) {

                        // even vector is 2d in nd4j
                        if (dims == 1)
                            dimensions.add(1);

                        for (int e = 0; e < dims; e++) {
                            // TODO: eventually we want long shapes :(
                            dimensions.add(getShapeFromAttr(shape)[e]);
                        }
                    }

                    arrayShape = Ints.toArray(dimensions);

                    variable.setShape(arrayShape);
                }

                if (attributes.containsKey(valueKey())) {
                    // value of?
                    ATTR_TYPE value = attributes.get(valueKey());

                    //DataType opType = value.

                    // log.debug("Dtype: {}", tensor.getDtype());

                    INDArray array = getArrayFrom(tfNode);
                    variable.setShape(array.shape());
                    variable.getSameDiff().associateArrayWithVariable(array,variable);
                }

                diff.addVariable(variable);
                graph.addVertex(vertex);
            }
        }


        return diff;
    }
}
