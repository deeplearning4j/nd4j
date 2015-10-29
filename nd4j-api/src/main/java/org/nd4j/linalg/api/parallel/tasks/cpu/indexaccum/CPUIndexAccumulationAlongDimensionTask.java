package org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.BaseTask;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.accumulation.CPUAccumulationTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.accumulation.CPUAccumulationViaTensorTask;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.RecursiveTask;


public class CPUIndexAccumulationAlongDimensionTask extends BaseCPUTask<INDArray> {
    protected final IndexAccumulation op;
    protected final int[] dimensions;

    protected List<Task<Pair<Double,Integer>>> subTasks;

    public CPUIndexAccumulationAlongDimensionTask(IndexAccumulation op, int parallelThreshold, int[] dimensions){
        super(op,parallelThreshold);
        this.op = op;
        this.dimensions = dimensions;
    }

    @Override
    public INDArray blockUntilComplete() {
        if(future == null){
            //invokeAsync() not called?
            invokeAsync();
        }

        INDArray out;
        try{
            out = future.get();
        }catch(Exception e){
            throw new RuntimeException(e);
        }
        if(out != null) return out; //FJ

        int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
        out = Nd4j.create(retShape);
        int i=0;
        for(Task<Pair<Double,Integer>> task : subTasks ){
            Pair<Double,Integer> result = task.blockUntilComplete();
            out.putScalar(i++,result.getSecond());
        }
        op.setZ(out);
        return out;
    }

    @Override
    public INDArray call() {
        int nTensors = op.x().tensorssAlongDimension(dimensions);
        subTasks = new ArrayList<>(nTensors);

        for( int i=0; i<nTensors; i++ ){
            IndexAccumulation opOnDimension = (IndexAccumulation)op.opForDimension(i,dimensions);
            INDArray x2 = opOnDimension.x();
            INDArray y2 = opOnDimension.y();

            boolean canDoDirectly;
            if(y2 == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
            else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

            Task<Pair<Double,Integer>> task;
            if(canDoDirectly){
                task = new CPUIndexAccumulationTask(opOnDimension,threshold,true);
            } else {
                task = new CPUIndexAccumulationViaTensorTask(op,threshold,true);
            }

            task.invokeAsync();
            subTasks.add(task);
        }
        return null;
    }

    @Override
    protected INDArray compute() {
        //Fork join
        INDArray x = op.x();
        INDArray y = op.y();
        int nTensors = x.tensorssAlongDimension(dimensions);
        List<RecursiveTask<Pair<Double,Integer>>> subTasks = new ArrayList<>(nTensors);
        
        if(dimensions.length == 1 && !op.isPassThrough() && x.ordering()=='c' && Arrays.equals(op.x().stride(), Nd4j.getStrides(x.shape(), x.ordering()))
                && (y==null || (y.ordering() == 'c' && Arrays.equals(op.y().stride(), Nd4j.getStrides(y.shape(), y.ordering())))) ){
            //Op along 1d -> need to calculate 1d tensor. Can use fast 1d tensor stats here
            OpExecutionerUtil.Tensor1DStats t1dx = OpExecutionerUtil.get1DTensorStats(x,dimensions[0]);
            int n = t1dx.getTensorLength();
            int ewsx = t1dx.getElementWiseStride();
            if(y == null ) {
                for (int i = 0; i < nTensors; i++) {
                    int offsetX = t1dx.getFirstTensorOffset() + i * t1dx.getTensorStartSeparation();
                    RecursiveTask<Pair<Double, Integer>> task = new CPUIndexAccumulationTask(op, threshold, n, offsetX, 0,
                            ewsx, 0, 0, true);
                    task.fork();
                    subTasks.add(task);
                }
            } else {
                OpExecutionerUtil.Tensor1DStats t1dy = OpExecutionerUtil.get1DTensorStats(y,dimensions[0]);
                int ewsy = t1dy.getElementWiseStride();
                for (int i = 0; i < nTensors; i++) {
                    int offsetX = t1dx.getFirstTensorOffset() + i * t1dx.getTensorStartSeparation();
                    int offsetY = t1dy.getFirstTensorOffset() + i * t1dy.getTensorStartSeparation();
                    RecursiveTask<Pair<Double, Integer>> task = new CPUIndexAccumulationTask(op, threshold, n, offsetX, offsetY,
                            ewsx, ewsy, 0, true);
                    task.fork();
                    subTasks.add(task);
                }
            }

        } else {
            for (int i = 0; i < nTensors; i++) {
                IndexAccumulation opOnDimension = (IndexAccumulation) op.opForDimension(i, dimensions);
                INDArray x2 = opOnDimension.x();
                INDArray y2 = opOnDimension.y();

                boolean canDoDirectly;
                if (y2 == null) canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2);
                else canDoDirectly = OpExecutionerUtil.canDoOpDirectly(x2, y2);

                RecursiveTask<Pair<Double, Integer>> task;
                if (canDoDirectly) {
                    task = new CPUIndexAccumulationTask(opOnDimension, threshold, true);
                } else {
                    task = new CPUIndexAccumulationViaTensorTask(op, threshold, true);
                }
                task.fork();
                subTasks.add(task);
            }
        }

        int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimensions);
        INDArray out = Nd4j.create(retShape);
        int i=0;
        for( RecursiveTask<Pair<Double,Integer>> task : subTasks ){
            Pair<Double,Integer> result = task.join();
            out.putScalar(i++,result.getSecond());
        }
        op.setZ(out);
        return out;
    }
}
