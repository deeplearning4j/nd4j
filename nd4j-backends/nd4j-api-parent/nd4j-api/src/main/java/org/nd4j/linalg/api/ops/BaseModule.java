package org.nd4j.linalg.api.ops;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

public abstract class BaseModule extends DynamicCustomOp implements Module {
    private List<Module> modules = new ArrayList<>();

    public BaseModule(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, List<Module> modules) {
        super(opName, inputs, outputs, tArguments, iArguments);
        this.modules = modules;
    }

    public BaseModule(String opName, SameDiff sameDiff, DifferentialFunction[] args, boolean inPlace, List<Module> modules) {
        super(opName, sameDiff, args, inPlace);
        this.modules = modules;
    }

    @Override
    public Module[] subModules() {
        return modules.toArray(new Module[modules.size()]);
    }

    @Override
    public void addModule(Module module) {
        modules.add(module);
    }




}
