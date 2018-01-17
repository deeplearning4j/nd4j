package org.nd4j.autodiff.gradcheck;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

@Slf4j
public class GradCheckMisc {

    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testMiscFunctions(){

        /*
        To test:
        tile
        reshape
        permute
        expandDims
        repeat
        rollAxis
        doRepeat
         */

    }

    @Test
    public void testReshapeGradient(){
        int[] origShape = new int[]{3,4,5};

        for( int[] toShape : new int[][]{{3,4*5}, {3*4,5}, {1,3*4*5}, {3*4*5,1}}){
            for(Pair<INDArray,String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, origShape)){
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable reshape = sd.reshape(in, toShape);
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", reshape,true );

                INDArray out = sd.execAndEndResult();
                INDArray expOut = in.getArr().std(true, Integer.MAX_VALUE);
                assertEquals(expOut, out);

                String msg = "toShape=" + Arrays.toString(toShape) + ", source=" + p.getSecond();
                boolean ok = GradCheckUtil.checkGradients(sd);
                assertTrue(msg, ok);
            }
        }
    }

    @Test
    public void testPermuteGradient(){
        int[] origShape = new int[]{3,4,5};

        for( int[] perm : new int[][]{{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}}){
            for(Pair<INDArray,String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, origShape)){
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable permute = sd.f().permute(in, perm);
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", permute,true );

                INDArray out = sd.execAndEndResult();
                INDArray expOut = in.getArr().std(true, Integer.MAX_VALUE);
                assertEquals(expOut, out);

                String msg = "permute=" + Arrays.toString(perm) + ", source=" + p.getSecond();
                boolean ok = GradCheckUtil.checkGradients(sd);
                assertTrue(msg, ok);
            }
        }
    }

    @Test
    public void testExpandDimsGradient(){
        int[] origShape = new int[]{3,4};

        for( int i=0; i<3; i++ ) {
            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAllTestMatricesWithShape(origShape[0], origShape[1], 12345)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable expand = sd.f().expandDims(in, i);
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", expand, true);

                INDArray out = sd.execAndEndResult();
                INDArray expOut = in.getArr().std(true, Integer.MAX_VALUE);
                assertEquals(expOut, out);

                String msg = "expandDim=" + i + ", source=" + p.getSecond();
                boolean ok = GradCheckUtil.checkGradients(sd);
                assertTrue(msg, ok);
            }
        }
    }

    @Test
    public void testSqueezeGradient(){
        int[] origShape = new int[]{3,4,5};

        for( int i=0; i<3; i++ ) {

            int[] shape = origShape.clone();
            shape[i] = 1;

            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, shape)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable squeeze = sd.f().squeeze(in, i);
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", squeeze, true);

                int[] expShapePostSqueeze;
                switch(i){
                    case 0:
                        expShapePostSqueeze = new int[]{4,5};
                        break;
                    case 1:
                        expShapePostSqueeze = new int[]{3,5};
                        break;
                    case 2:
                        expShapePostSqueeze = new int[]{3,4};
                        break;
                    default:
                        throw new RuntimeException();
                }

                sd.execAndEndResult();

                INDArray squeezed = squeeze.getArr();
                assertArrayEquals(expShapePostSqueeze, squeezed.shape());

                INDArray out = sd.execAndEndResult();
                INDArray expOut = in.getArr().std(true, Integer.MAX_VALUE);
                assertEquals(expOut, out);

                String msg = "squeezeDim=" + i + ", source=" + p.getSecond();
                boolean ok = GradCheckUtil.checkGradients(sd);
                assertTrue(msg, ok);
            }
        }
    }

    @Test
    public void testGradientAutoBroadcast1(){

        Nd4j.getRandom().setSeed(12345);

        List<String> allFailed = new ArrayList<>();

        for (int dim_sz1 : new int[]{0, 1, 2}) {

            int[] in2Shape = {3,4,5};
            in2Shape[dim_sz1] = 1;

            for (int i = 0; i < 6; i++) {

                SameDiff sd = SameDiff.create();

                SDVariable in3 = sd.var("in3", Nd4j.rand(new int[]{3,4,5}));
                SDVariable in2 = sd.var("in2", in2Shape);

                SDVariable bcOp;
                switch (i) {
                    case 0:
                        bcOp = in3.add(in2);
                        break;
                    case 1:
                        bcOp = in3.sub(in2);
                        break;
                    case 2:
                        bcOp = in3.mul(in2);
                        break;
                    case 3:
                        bcOp = in3.div(in2);
                        break;
                    case 4:
                        bcOp = in3.rsub(in2);
                        break;
                    case 5:
                        bcOp = in3.rdiv(in2);
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable outVar = sd.sum(bcOp);

                String msg = "(test " + i + ", dimension=" + dim_sz1 + ")";
                log.info("*** Starting test: " + msg);

                INDArray in3Arr = Nd4j.randn(new int[]{3,4,5}).muli(100);
                INDArray in2Arr = Nd4j.randn(in2Shape).muli(100);

                sd.associateArrayWithVariable(in3Arr, in3);
                sd.associateArrayWithVariable(in2Arr, in2);

                try {
                    INDArray out = sd.execAndEndResult();
                    assertNotNull(out);
                    assertArrayEquals(new int[]{1, 1}, out.shape());

//                    System.out.println(sd.asFlatPrint());

                    boolean ok = GradCheckUtil.checkGradients(sd);
                    if(!ok){
                        allFailed.add(msg);
                    }
                } catch (Exception e){
                    e.printStackTrace();
                    allFailed.add(msg + " - EXCEPTION");
                }
            }
        }

        assertEquals("Failed: " + allFailed, 0, allFailed.size());
    }

    @Test
    public void testGradientAutoBroadcast2(){

        Nd4j.getRandom().setSeed(12345);

        List<String> allFailed = new ArrayList<>();

        for (int[] dim_sz1s : new int[][]{{0,1}, {0,2}, {1,2}}) {

            int[] in1Shape = {3,4,5};
            in1Shape[dim_sz1s[0]] = 1;
            in1Shape[dim_sz1s[1]] = 1;

            for (int i = 0; i < 6; i++) {

                SameDiff sd = SameDiff.create();

                int nOut = 4;
                int minibatch = 10;
                SDVariable in3 = sd.var("in3", Nd4j.rand(new int[]{3,4,5}));
                SDVariable in2 = sd.var("in2", in1Shape);

                SDVariable bcOp;
                switch (i) {
                    case 0:
                        bcOp = in3.add(in2);
                        break;
                    case 1:
                        bcOp = in3.sub(in2);
                        break;
                    case 2:
                        bcOp = in3.mul(in2);
                        break;
                    case 3:
                        bcOp = in3.div(in2);
                        break;
                    case 4:
                        bcOp = in3.rsub(in2);
                        break;
                    case 5:
                        bcOp = in3.rdiv(in2);
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable outVar = sd.sum(bcOp);

                String msg = "(test " + i + ", dimensions=" + Arrays.toString(dim_sz1s) + ")";
                log.info("*** Starting test: " + msg);

                INDArray in3Arr = Nd4j.randn(minibatch, nOut).muli(100);
                INDArray in2Arr = Nd4j.randn(minibatch, nOut).muli(100);

                sd.associateArrayWithVariable(in3Arr, in3);
                sd.associateArrayWithVariable(in2Arr, in2);

                try {
                    INDArray out = sd.execAndEndResult();
                    assertNotNull(out);
                    assertArrayEquals(new int[]{1, 1}, out.shape());

//                    System.out.println(sd.asFlatPrint());

                    boolean ok = GradCheckUtil.checkGradients(sd);
                    if(!ok){
                        allFailed.add(msg);
                    }
                } catch (Exception e){
                    e.printStackTrace();
                    allFailed.add(msg + " - EXCEPTION");
                }
            }
        }

        assertEquals("Failed: " + allFailed, 0, allFailed.size());
    }

    @Test
    public void testSlice(){
        Nd4j.getRandom().setSeed(12345);

        //Order here: original shape, begin, size
        List<Triple<int[],int[], int[]>> testCases = new ArrayList<>();
        testCases.add(new Triple<>(new int[]{3,4},new int[]{0,0}, new int[]{3,4}));
        testCases.add(new Triple<>(new int[]{3,4},new int[]{1,1}, new int[]{3,4}));
        testCases.add(new Triple<>(new int[]{3,4},new int[]{1,2}, new int[]{2,3}));
        testCases.add(new Triple<>(new int[]{3,4,5},new int[]{0,0,0}, new int[]{3,4,5}));
        testCases.add(new Triple<>(new int[]{3,4,5},new int[]{1,1,1}, new int[]{2,3,4}));
        testCases.add(new Triple<>(new int[]{3,4,5},new int[]{1,0,2}, new int[]{3,3,4}));

        for( int i=0; i<testCases.size(); i++ ){
            Triple<int[],int[], int[]> t = testCases.get(i);
            int[] os = t.getFirst();
            int[] b = t.getSecond();
            int[] e = t.getThird();
            INDArray arr = Nd4j.rand(os);

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", arr);
            SDVariable slice = sd.slice(in, b, e);
            SDVariable stdev = sd.standardDeviation(slice, true);

            String msg = "i=" + i + ": inShape=" + Arrays.toString(os) + ", begin=" + Arrays.toString(b) + ", end=" + Arrays.toString(e);
            log.info("Starting test: " + msg);
            GradCheckUtil.checkGradients(sd);
        }
    }


    @Builder(builderClassName = "Builder")
    @Data
    private static class SSCase {
        private int[] shape;
        private int[] begin;
        private int[] end;
        private int[] strides;
        private int beginMask;
        private int endMask;
        private int ellipsisMask;
        private int newAxisMask;
        private int shrinkAxisMask;

        public static class Builder {

            public Builder shape(int... shape){
                this.shape = shape;
                return this;
            }

            public Builder begin(int... begin){
                this.begin = begin;
                return this;
            }

            public Builder end(int... end){
                this.end = end;
                return this;
            }

            public Builder strides(int... strides){
                this.strides = strides;
                return this;
            }
        }
    }

    @Test
    public void testStridedSlice(){
        Nd4j.getRandom().setSeed(12345);

        //Order here: original shape, begin, size
        List<SSCase> testCases = new ArrayList<>();
        testCases.add(SSCase.builder().shape(3,4).begin(0,0).end(3,4).strides(1,1).build());
        testCases.add(SSCase.builder().shape(3,4).begin(1,1).end(2,3).strides(1,1).build());
        testCases.add(SSCase.builder().shape(3,4).begin(-999,0).end(3,4).strides(1,1).beginMask(1).build());
        testCases.add(SSCase.builder().shape(3,4).begin(1,1).end(3,-999).strides(1,1).endMask(1<<1).build());
        testCases.add(SSCase.builder().shape(3,4).begin(-999,0).end(-999,4).strides(1,1).beginMask(1).endMask(1).build());
        testCases.add(SSCase.builder().shape(3,4).begin(-999,0,0).end(-999,3,4).strides(1,1).newAxisMask(1).build());

        testCases.add(SSCase.builder().shape(3,4,5).begin(0,0,0).end(3,4,5).strides(1,1,1).build());
        testCases.add(SSCase.builder().shape(3,4,5).begin(1,2,3).end(3,4,5).strides(1,1,1).build());
        testCases.add(SSCase.builder().shape(3,4,5).begin(0,0,0).end(3,3,5).strides(1,2,2).build());
        testCases.add(SSCase.builder().shape(3,4,5).begin(1,-999,1).end(3,3,4).strides(1,1,1).beginMask(1<<1).build());
        testCases.add(SSCase.builder().shape(3,4,5).begin(1,-999,1).end(3,3,-999).strides(1,1,1).beginMask(1<<1).endMask(1<<2).build());
        testCases.add(SSCase.builder().shape(3,4,5).begin(1,2).end(3,4).strides(1,1).ellipsisMask(1<<1).build());   //[1:3,...,2:4]
        testCases.add(SSCase.builder().shape(3,4,5).begin(1,-999,1,2).end(3,-999,3,4).strides(1,-999,1,2).newAxisMask(1<<1).build());
        testCases.add(SSCase.builder().shape(3,4,5).begin(1,0,1).end(3,-999,4).strides(1,1,1).shrinkAxisMask(1<<1).build());
        testCases.add(SSCase.builder().shape(3,4,5).begin(1,1,1).end(3,-999,4).strides(1,1,1).shrinkAxisMask(1<<1).build());



        for( int i=0; i<testCases.size(); i++ ){
            SSCase t = testCases.get(i);
            INDArray arr = Nd4j.rand(t.getShape());

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", arr);
            SDVariable slice = sd.stridedSlice(in, t.getBegin(), t.getEnd(), t.getStrides(), t.getBeginMask(),
                    t.getEndMask(), t.getEllipsisMask(), t.getNewAxisMask(), t.getShrinkAxisMask());
            SDVariable stdev = sd.standardDeviation(slice, true);

            String msg = "i=" + i + ": " + t;
            log.info("Starting test: " + msg);
            GradCheckUtil.checkGradients(sd);
        }
    }
}
