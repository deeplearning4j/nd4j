package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;


public class ReverseSequence extends DynamicCustomOp {


    int seqDim;
    int batchDim;



    public ReverseSequence(SameDiff sameDiff, SDVariable i_v, SDVariable seqLengths, int seqDim, int batchDim) {
        super(null, sameDiff, new SDVariable[]{i_v, seqLengths}, false);

        this.seqDim = seqDim;
        this.batchDim = batchDim;
        addArguments();

    }

    public ReverseSequence(SameDiff sameDiff, SDVariable i_v, SDVariable seqLengths) {
        super(null, sameDiff, new SDVariable[]{i_v, seqLengths}, false);
        this.seqDim = 1;
        this.batchDim = 0;
        addArguments();
    }

    private void addArguments(){
        addIArgument(seqDim);
        addIArgument(batchDim);
    }

    public ReverseSequence() {
    }

    @Override
    public String opName() {
        return "reverse_sequense";

    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("ReverseSequence");
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable ret = f().reverse_sequence(f1.get(0), f1.get(1), seqDim, batchDim);
        return Arrays.asList(ret);
    }

}
