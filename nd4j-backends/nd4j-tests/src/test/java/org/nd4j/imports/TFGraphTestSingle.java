package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.Map;

import static org.nd4j.imports.TFGraphTestAll.inputVars;
import static org.nd4j.imports.TFGraphTestAll.outputVars;
import static org.nd4j.imports.TFGraphTestAll.testSingle;

/**
 * Created by susaneraly on 11/12/17.
 */
@Slf4j
public class TFGraphTestSingle {

    @Test
    public void testOne() throws  Exception {
        //String modelName = "transform_0";
        String modelName = "conv_0";
        String modelDir = new ClassPathResource("tf_graphs/examples/" + modelName).getFile().getAbsolutePath();
        Map<String, INDArray> inputs = inputVars(modelDir);
        Map<String, INDArray> predictions = outputVars(modelDir);
        testSingle(inputs,predictions,modelName,modelDir);
    }
}
