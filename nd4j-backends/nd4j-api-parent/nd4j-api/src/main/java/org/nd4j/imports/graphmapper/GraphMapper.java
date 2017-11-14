package org.nd4j.imports.graphmapper;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

/**
 * Map graph proto types to
 *
 * {@link SameDiff} instances
 * @param <GRAPH_TYPE> the proto type for the graph
 * @param <NODE_TYPE> the proto type for the node
 * @param <ATTR_TYPE> the proto tyoe fpr the attribute
 *
 *@author Adam Gibson
 */
public interface GraphMapper<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE> {



    /**
     * The attribute key for data type
     * @return
     */
    String valueKey();

    /**
     * The attribute key for data type
     * @return
     */
    String shapeKey();

    /**
     * The attribute key for data type
     * @return
     */
    String dTypeKey();

    /**
     * Get the shape of the attribute value
     * @param attr the attribute value
     * @return the shape of the attribute if any or null
     */
    int[] getShapeFromAttr(ATTR_TYPE attr);

    /**
     * Get the attribute
     * map for given node
     * @param nodeType the node
     * @return the attribute map for the attribute
     */
    Map<String,ATTR_TYPE> getAttrMap(NODE_TYPE nodeType);

    /**
     * Get the name of the node
     * @param nodeType the node
     *                 to get the name for
     * @return
     */
    String getName(NODE_TYPE nodeType);

    /**
     *
     * @param nodeType
     * @return
     */
    boolean alreadySeen(NODE_TYPE nodeType);

    /**
     *
     * @param nodeType
     * @return
     */
    boolean isVariableNode(NODE_TYPE nodeType);

    /**
     *
     *
     * @param opType
     * @return
     */
    boolean shouldSkip(NODE_TYPE opType);

    /**
     *
     * @param nodeType
     * @return
     */
    boolean hasShape(NODE_TYPE nodeType);

    /**
     *
     * @param nodeType
     * @return
     */
    int[] getShape(NODE_TYPE nodeType);

    /**
     *
     * @param nodeType
     * @return
     */
    INDArray getArrayFrom(NODE_TYPE nodeType);


    /**
     *
     * @param graphType
     * @return
     */
    List<NODE_TYPE> getNodeList(GRAPH_TYPE graphType);

    /**
     * This method converts given TF
     * @param tfGraph
     * @return
     */
     SameDiff importGraph(GRAPH_TYPE tfGraph);

}
