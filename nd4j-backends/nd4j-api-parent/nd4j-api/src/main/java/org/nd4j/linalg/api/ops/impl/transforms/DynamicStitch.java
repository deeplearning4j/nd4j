package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


/**
 * Transforms a given input tensor into numPartitions partitions, as indicated by the indices in "partitions".
 * Output tensor has one more dimension than input tensor, the first dimension indicates the partition.
 * <p>
 * Example:
 * <p>
 * input:           [4, 3, 5, 7, 8, 0]
 * input shape:     [1, 6]
 * partitions:      [1, 0, 1, 0, 0, 1]
 * numPartitions:   2
 * outputs[0]:      [3, 7, 8]
 * outputs[1]:      [4, 5, 0]
 *
 * @author Max Pumperla
 */
public class DynamicStitch extends DynamicCustomOp {

    private int numPartitions;
    private SDVariable indices;

    public DynamicStitch() {
    }

    public DynamicStitch(SameDiff sameDiff, SDVariable[] inputAndpartitions) {
        super(null, sameDiff, inputAndpartitions, false);

        SDVariable input = inputAndpartitions[0];
        SDVariable sdPartitions = inputAndpartitions[1];

        this.indices = sdPartitions;
        this.numPartitions = input.getArr().shape()[0];
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // DynamicPartition and DynamicStitch are mutually inverse
        SDVariable gradient = i_v.get(0);
        SDVariable ret = sameDiff.dynamicPartition(gradient, indices, numPartitions);
        return Collections.singletonList(ret);
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
    }


    @Override
    public String opName() {
        return "dynamic_stitch";
    }


    @Override
    public String tensorflowName() {
        return "DynamicStitch";
    }

    @Override
    public String onnxName() {
        throw new IllegalStateException("Dynamic partitioning currently not supported by ONNX");
    }

}
