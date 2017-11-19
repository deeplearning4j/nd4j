package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.tensorflow.framework.GraphDef;

import java.io.File;

/**
 * This class provides TensorFlow graphs & models import
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class TensorFlowImport {

    /**
     *
     * @param graphFile
     * @return
     */
    public static SameDiff importGraph(File graphFile) {
        return new TFGraphMapper().importGraph(graphFile);
    }

    /**
     * This method converts given TF
     * @param tfGraph
     * @return
     */
    public static SameDiff importGraph(GraphDef tfGraph) {
        return new TFGraphMapper().mapGraph(tfGraph);
    }

}
