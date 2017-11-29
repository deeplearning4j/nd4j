package org.nd4j.imports;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.Map;

import static org.nd4j.imports.TFGraphTestAllHelper.inputVars;
import static org.nd4j.imports.TFGraphTestAllHelper.outputVars;
import static org.nd4j.imports.TFGraphTestAllHelper.testSingle;

/**
 * TFGraphTestAll will run all the checked in TF graphs and
 * compare outputs in nd4j to those generated and checked in from TF.
 * <p>
 * This file is to run a single graph or a list of graphs that are checked in to aid in debug.
 * Simply change the modelNames String[] in testSome() to correspond to the directory name the graph lives in
 * - eg. to run the graph for 'bias_add' i.e checked in under tf_graphs/examples/bias_add
 * set modelNames to "bias_add"
 */
public class TFGraphTestList {

    public static String modelDir = TFGraphTestAllHelper.COMMON_BASE_DIR; //this is for later if we want to check in models separately for samediff and libnd4j
    public static String[] modelNames = new String[]{
            //"add_n",
            //"ae_00",
            "bias_add",
            //"conv_0",
            //"g_00",
            //"g_01",
            //"math_mul_order",
            //"mlp_00",
            //"transform_0",
            //"transpose"
            };
    //change this to SAMEDIFF for samediff
    //public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.SAMEDIFF;
    public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.LIBND4J;

    @Test
    public void testSome() throws IOException {
        for (int i = 0; i < modelNames.length; i++) {
            Map<String, INDArray> inputs = inputVars(modelNames[i] , modelDir);
            Map<String, INDArray> predictions = outputVars(modelNames[i], modelDir);
            testSingle(inputs, predictions, modelNames[i], modelDir, executeWith);
        }
    }
}
