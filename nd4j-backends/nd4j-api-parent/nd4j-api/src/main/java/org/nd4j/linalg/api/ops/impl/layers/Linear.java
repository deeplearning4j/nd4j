package org.nd4j.linalg.api.ops.impl.layers;

import lombok.Builder;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseModule;
import org.nd4j.linalg.api.ops.Module;
import org.nd4j.linalg.api.ops.impl.accum.Mmul;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Linear:
 * a * bT
 *
 * @author Adam Gibson
 */
public class Linear extends BaseModule {

    private Mmul forward;
    private int nIn,nOut;

    @Builder(builderMethodName = "execBuilder")
    public Linear(int nIn,int nOut) {
        super(null, new INDArray[]{Nd4j.create(nIn)},new INDArray[]{Nd4j.create()}, new ArrayList<Double>(), new ArrayList<Integer>(),new ArrayList<Module>());
    }

    @Builder(builderMethodName = "sameDiffBuilder")
    public Linear(SameDiff sameDiff, DifferentialFunction[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace, new ArrayList<Module>());
    }

    @Override
    public String opName() {
        return "linear";
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        execSameDiff();
        return forward.doDiff(f1);
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>();
        ret.add(Shape.getMatrixMultiplyShape(getInputArguments().get(0).shape(),getInputArguments().get(1).transpose().shape()));
        return ret;
    }

    @Override
    public void exec() {
        if(this.getInputArguments().isEmpty()) {
            throw new IllegalStateException("No arguments found.");
        }

        INDArray input = getInputArguments().get(0);
        INDArray right = getInputArguments().get(1);
        if(getOutputArguments().isEmpty()) {
            getOutputArguments().add(input.mmul(right.transpose()));
        }
        else {
            input.mmul(right.transpose(),getOutputArguments().get(0));
        }

    }

    @Override
    public void execSameDiff() {
        if(args == null || args.length == 0) {
            throw new IllegalStateException("No arguments found");
        }

        if(forward == null) {
            forward = new Mmul(sameDiff, args()[0], args()[1],
                    MMulTranspose.builder().transposeA(false).transposeB(true).build());
            this.outputFunctions = forward.outputFunctions();
        }


    }
}
