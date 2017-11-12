package org.nd4j.linalg.api.ops;

import org.nd4j.autodiff.functions.DifferentialFunction;

import java.util.List;

public class DefaultOpConverter extends BaseOp {
    private static DefaultOpConverter INSTANCE = new DefaultOpConverter();
    public static DefaultOpConverter getInstance() {
        return INSTANCE;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "defaultop";
    }
}
