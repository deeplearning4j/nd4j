package org.nd4j.imports;

import lombok.val;
import org.junit.Test;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.linalg.api.ops.impl.controlflow.Gather;
import org.nd4j.linalg.io.ClassPathResource;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class OnnxImportTest {

    @Test
    public void testOnnxImportEmbedding() throws Exception {
        /**
         *
         */
        val importGraph = OnnxGraphMapper.getInstance().importGraph(new ClassPathResource("onnx_graphs/embedding_only.onnx").getFile());
        assertEquals(3,importGraph.graph().numVertices());
        val embeddingMatrix = importGraph.getVariable("2");
        assertArrayEquals(new int[] {100,300},embeddingMatrix.getShape());
        val onlyOp = importGraph.getFunctionForVertexId(importGraph.getVariable("3").getVertexId());
        assertNotNull(onlyOp);
        assertTrue(onlyOp instanceof Gather);

    }

    @Test
    public void testOnnxImportCnn() throws Exception {
        val importGraph = OnnxGraphMapper.getInstance().importGraph(new ClassPathResource("onnx_graphs/sm_cnn.onnx").getFile());
        assertEquals(12,importGraph.graph().numVertices());
        System.out.println(importGraph);

    }

}
