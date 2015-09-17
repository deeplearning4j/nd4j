package org.nd4j.linalg.shape.loop;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.loop.coordinatefunction.CoordinateFunction;
import org.nd4j.linalg.api.shape.loop.two.LoopFunction2;
import org.nd4j.linalg.api.shape.loop.two.RawArrayIterationInformation2;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Created by agibsonccc on 9/15/15.
 */
public class LoopTests extends BaseNd4jTest {
    @Test
    public void testLoop2d() {
        INDArray arr = Nd4j.linspace(1,6,6).reshape(2,3);
        INDArray arr2 = Nd4j.rand(2,3);
        Shape.assignArray(arr2,arr);
        assertEquals(arr,arr2);
        System.out.println(arr2);
    }




    @Override
    public char ordering() {
        return 'c';
    }
}
