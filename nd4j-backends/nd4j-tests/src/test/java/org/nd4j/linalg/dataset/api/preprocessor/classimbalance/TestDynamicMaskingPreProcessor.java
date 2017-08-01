package org.nd4j.linalg.dataset.api.preprocessor.classimbalance;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by Alex on 09/06/2017.
 */
public class TestDynamicMaskingPreProcessor extends BaseNd4jTest {

    public TestDynamicMaskingPreProcessor(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testDynamicMaskingPreProcessor() {

        //Start with edge cases: 0 and 1
        DynamicMaskingPreProcessor ppBinary00 = new DynamicMaskingPreProcessor(DynamicMaskingPreProcessor.OutputType.Binary,
                Nd4j.zeros(2));
        DynamicMaskingPreProcessor ppBinary11 = new DynamicMaskingPreProcessor(DynamicMaskingPreProcessor.OutputType.Binary,
                Nd4j.ones(2));
        DynamicMaskingPreProcessor ppBinary01 = new DynamicMaskingPreProcessor(DynamicMaskingPreProcessor.OutputType.Binary,
                Nd4j.create(new double[]{0.0, 1.0}));
        DynamicMaskingPreProcessor ppBinary10 = new DynamicMaskingPreProcessor(DynamicMaskingPreProcessor.OutputType.Binary,
                Nd4j.create(new double[]{1.0, 0.0}));

        INDArray l = Nd4j.valueArrayOf(10,1, 10);
        DataSet dsA = new DataSet(null, l.dup());

        //Expect no change with 0 probability of masking
        ppBinary00.preProcess(dsA);
        assertEquals(l, dsA.getLabels());
        assertEquals(Nd4j.ones(10,1), dsA.getLabelsMaskArray());
        dsA.setLabelsMaskArray(null);

        //Expect all to be masked with 1 probability of masking
        ppBinary11.preProcess(dsA);
        assertEquals(l, dsA.getLabels());
        assertEquals(Nd4j.zeros(10,1), dsA.getLabelsMaskArray());

        l = Nd4j.create(new double[]{0,0,0,0,0,1,1,1,1,1}, new int[]{10,1});
        DataSet dsB = new DataSet(null, l.dup());

        INDArray expMask01 = Nd4j.create(new double[]{0,0,0,0,0,1,1,1,1,1}, new int[]{10,1});
        INDArray expMask10 = Nd4j.create(new double[]{1,1,1,1,1,0,0,0,0,0}, new int[]{10,1});

        //Expect only class 0 to be masked with 0,1 probability
        ppBinary01.preProcess(dsB);
        assertEquals(l, dsB.getLabels());
        assertEquals(expMask01, dsB.getLabelsMaskArray());
        dsB.setLabelsMaskArray(null);

        //Expect only class 1 to be masked with 1,0 probability
        ppBinary10.preProcess(dsB);
        assertEquals(l, dsB.getLabels());
        assertEquals(expMask10, dsB.getLabelsMaskArray());

    }

    @Test
    public void testDynamicMaskingMultiPreProcessor() {

        fail();
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
