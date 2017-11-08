package org.nd4j.linalg.api.ops.impl.controlflow;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class IfDerivative extends If {

    private If ifDelegate;

    public IfDerivative(If ifBlock) {
        super(ifBlock);
        this.ifDelegate = ifBlock;
    }

    @Override
    public Boolean getTrueBodyExecuted() {
        return ifDelegate.trueBodyExecuted;
    }


    @Override
    public SameDiff.SameDiffFunctionDefinition getFalseBody() {
        return ifDelegate.falseBody;
    }

    @Override
    public SameDiff getFalseBodyExecution() {
        return ifDelegate.falseBodyExecution;
    }

    @Override
    public String getBlockName() {
        return ifDelegate.blockName;
    }

    @Override
    public String getFalseBodyName() {
        return ifDelegate.falseBodyName;
    }

    @Override
    public SameDiff getLoopBodyExecution() {
        return ifDelegate.loopBodyExecution;
    }

    @Override
    public SameDiff.SameDiffConditional getPredicate() {
        return ifDelegate.getPredicate();
    }

    @Override
    public SameDiff getPredicateExecution() {
        return ifDelegate.predicateExecution;
    }

    @Override
    public List<int[]> calculateOutputShape() {
        return super.calculateOutputShape();
    }

    @Override
    public String opName() {
        return "if_bp";
    }

    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
        throw new UnsupportedOperationException("Unable to take the derivative of the derivative for if");
    }
}
