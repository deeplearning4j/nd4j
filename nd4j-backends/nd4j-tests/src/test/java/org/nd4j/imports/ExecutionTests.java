package org.nd4j.imports;

import lombok.val;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.Nd4jCpu;

import static org.junit.Assert.assertEquals;

/**
 * This set of tests suited for validation of various graph execuction methods: flatbuffers, stored graphs reuse, one-by-one execution, etc
 *
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class ExecutionTests extends BaseNd4jTest {

    public ExecutionTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testStoredGraph_1()  throws Exception {
        Nd4j.create(1);

        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/reduce_dim.pb.txt").getInputStream());
        val bb = tg.asFlatBuffers();

        val ptr = new BytePointer(bb);

        NativeOpsHolder.getInstance().getDeviceNativeOps().registerGraphFloat(null, 119, ptr);

        val array = Nd4j.create(3, 3).assign(3);

        val ptrBuffers = new PointerPointer(32);
        val ptrShapes = new PointerPointer(32);
        val ptrIndices = new IntPointer(new int[] {1});

        ptrBuffers.put(0, array.data().addressPointer());
        ptrShapes.put(0, array.shapeInfoDataBuffer().addressPointer());


        val backP = (Nd4jCpu.FloatVariablesSet) NativeOpsHolder.getInstance().getDeviceNativeOps().executeStoredGraphFloat(null, 119, ptrBuffers, ptrShapes, ptrIndices, 1);

        assertEquals(0, backP.status());

    }

    @Override
    public char ordering() {
        return 'c';
    }
}
