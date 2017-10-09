package org.nd4j.linalg.api.ops;

import lombok.Getter;
import lombok.val;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Basic implementation for CustomOp
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DynamicCustomOp extends DifferentialFunction implements CustomOp {

    private String opName;
    @Getter private List<INDArray> inputArguments;
    @Getter private List<INDArray> outputArguments;
    @Getter private List<Double> tArguments = new ArrayList<>();
    @Getter private List<Integer> iArguments = new ArrayList<>();
    @Getter private boolean inplaceCall;
    @Getter private long hash;

    public DynamicCustomOp() {
    }

    public DynamicCustomOp(String opName, SameDiff sameDiff, DifferentialFunction[] args) {
        super(sameDiff, args);
        this.opName = opName;
    }

    /**
     * Initialize this custom op with all of the
     * inputs, outputs, and respective
     * argumentts for execution
     * @param opName the name of the op to execute
     * @param inputs the inputs to the op
     * @param outputs the outputs of the op
     * @param tArguments the input float arguments
     * @param iArguments the input int arguments
     */
    public DynamicCustomOp(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments) {
        inputArguments = new ArrayList<>(Arrays.asList(inputs));
        outputArguments = new ArrayList<>(Arrays.asList(outputs));
        this.opName = opName;
        this.tArguments = tArguments;
        this.iArguments = iArguments;
    }


    /**
     * Initialize this operation for execution (pre created ndarrays)
     * @param opName the operation name to use
     *               for invocation
     * @param inputs the inputs
     * @param outputs the outputs of the op
     */
    public DynamicCustomOp(String opName,INDArray[] inputs,INDArray[] outputs) {
        this(opName,inputs,outputs, Collections.<Double>emptyList(),Collections.<Integer>emptyList());
    }

    /**
     * Initialize this for {@link SameDiff} execution
     * Any extra int or float arguments for operations
     * must be added to the respective {@link #getTArguments()}
     *  or {@link #getIArguments()} lists upon construction
     * @param opName the operation name
     * @param sameDiff the samediff instance to use
     * @param args the arguments to use
     * @param inPlace whether the operation is in place or not
     *
     */
    public DynamicCustomOp(String opName,SameDiff sameDiff, DifferentialFunction[] args, boolean inPlace) {
        super(sameDiff, inPlace, args);
        this.opName = opName;
        iArguments = new ArrayList<>();
        tArguments = new ArrayList<>();
    }

    protected DynamicCustomOp(String opName) {
        this.opName = opName;
    }

    /**
     * This method returns op name as string
     *
     * @return
     */
    @Override
    public String opName() {
        return opName;
    }

    /**
     * This method returns LongHash of the opName()
     *
     * @return
     */
    @Override
    public long opHash() {
        return hash;
    }

    /**
     * This method takes custom opname, and return Op DynamicCustomOpsBuilder instance
     * @param opName
     * @return
     */
    public static DynamicCustomOpsBuilder builder(String opName) {
        val map = Nd4j.getExecutioner().getCustomOperations();
        val lcName = opName.toLowerCase();
        val desc = map.get(lcName);

        if (desc == null)
            throw new ND4JIllegalStateException("Unknown operations requested: [" + opName + "]");

        return new DynamicCustomOpsBuilder(opName, desc.getHash(), desc.getNumInputs(), desc.getNumOutputs(), desc.isAllowsInplace(), desc.getNumTArgs(), desc.getNumIArgs());
    }

    @Override
    public ArrayField doGetValue() {
        throw new UnsupportedOperationException("Please extend DynamicCustomOp to run samediff graph operations.");
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Please extend DynamicCustomOp to run samediff graph operations.");
    }

    @Override
    public String toString() {
        return opName();
    }


    public static class SameDiffBuilder extends DynamicCustomOpsBuilder {
        private SameDiff sameDiff;
        private DifferentialFunction[] args;
        protected SameDiffBuilder(String opName, SameDiff sameDiff,DifferentialFunction[] args, long hash, int numInputs, int numOutputs, boolean inplaceAllowed, int numTArguments, int numIArguments) {
            super(opName, hash, numInputs, numOutputs, inplaceAllowed, numTArguments, numIArguments);
            this.args = args;
            this.sameDiff = sameDiff;
        }

        @Override
        public DynamicCustomOp build() {
            DynamicCustomOp ret =  super.build();
            ret.setArgs(args);
            ret.setSameDiff(sameDiff);
            return ret;
        }
    }

    public static class DynamicCustomOpsBuilder {
        protected String opName;
        protected int numInputs;
        protected int numOutputs;
        protected int numTArguments;
        protected int numIArguments;
        protected boolean inplaceCall;
        protected boolean inplaceAllowed;
        protected long opHash;

        private List<INDArray> inputArguments = new ArrayList<>();
        private List<INDArray> outputArguments = new ArrayList<>();
        private List<Double> tArguments = new ArrayList<>();
        private List<Integer> iArguments = new ArrayList<>();

        protected DynamicCustomOpsBuilder(String opName, long hash, int numInputs, int numOutputs, boolean inplaceAllowed, int numTArguments, int numIArguments) {
            this.opHash = hash;
            this.opName = opName;
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            this.numIArguments = numIArguments;
            this.numTArguments = numTArguments;
            this.inplaceAllowed = inplaceAllowed;
        }

        /**
         * This method
         * takes arbitrary number of input INDArrays in, as Op input
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate lengths/shapes.
         *
         * @param inputs
         * @return
         */
        public DynamicCustomOpsBuilder addInputs(INDArray... inputs) {
            // if we have positive value as numInputs - we should ensure equal amount of arguments
            if (numInputs >= 0) {
                if (inputs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numInputs + " arguments. Null was passed instead.");

                if (numInputs != inputs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numInputs + " arguments, but " + inputs.length + " was passed to constructor");
            }

            for (val in: inputs)
                inputArguments.add(in);

            return this;
        }

        /**
         * This method takes arbitrary number of 
         * output INDArrays in, to store operation result
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate lengths/shapes.
         *
         * @param outputs
         * @return
         */
        public DynamicCustomOpsBuilder addOutputs(INDArray... outputs) {
            if (numOutputs >= 0) {
                if (outputs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numOutputs + " arguments. Null was passed instead.");

                if (numOutputs != outputs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numOutputs + " arguments, but " + outputs.length + " was passed to constructor");
            }

            for (val in: outputs)
                outputArguments.add(in);

            return this;
        }

        /**
         * Whether an op call is in place or not.
         * @param reallyCall
         * @return
         */
        public DynamicCustomOpsBuilder callInplace(boolean reallyCall) {
            if (reallyCall && !inplaceAllowed)
                throw new ND4JIllegalStateException("Requested op can't be called inplace");

            this.inplaceCall = reallyCall;
            return this;
        }

        /**
         * This method takes arbitrary number of Integer arguments for op,
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @param iargs
         * @return
         */
        public DynamicCustomOpsBuilder addIntegerArguments(Integer... iargs) {
            if (numIArguments >= 0) {
                if (iargs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments. Null was passed instead.");

                if (numIArguments != iargs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments, but " + iargs.length + " was passed to constructor");
            }

            for (val in: iargs)
                iArguments.add(in);

            return this;
        }

        /**
         * This method takes arbitrary number of Integer arguments for op,
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @param arg
         * @return
         */
        public DynamicCustomOpsBuilder addIntegerArguments(int arg) {
            if (numIArguments != 1 && numIArguments > 0)
                throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments. One arg was passed instead.");

            iArguments.add(arg);

            return this;
        }

        /**
         * This method takes arbitrary number of Integer arguments for op,
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @param iargs
         * @return
         */
        public DynamicCustomOpsBuilder addIntegerArguments(int... iargs) {
            if (numIArguments >= 0) {
                if (iargs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments. Null was passed instead.");

                if (numIArguments != iargs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments, but " + iargs.length + " was passed to constructor");
            }

            for (val in: iargs)
                iArguments.add(in);

            return this;
        }

        /**
         * This method takes arbitrary number of Double arguments for op,
         * Note that this ACCUMULATES arguments. You are able to call this method
         * multiple times and it will add arguments to a list.
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @return
         */
        public DynamicCustomOpsBuilder addFloatingPointArguments(Double... targs) {
            if (numTArguments >= 0) {
                if (targs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numTArguments + " integer arguments. Null was passed instead.");

                if (numTArguments != targs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numTArguments + " integer arguments, but " + targs.length + " was passed to constructor");
            }

            for (val in: targs)
                tArguments.add(in);

            return this;
        }





        public DynamicCustomOp build() {
            // Eventually we probably will lift this restriction
            if (!inplaceCall && outputArguments.size() == 0)
                throw new ND4JIllegalStateException("If operation is not-inplace, it must have outputs defined");

            val result = new DynamicCustomOp(opName);
            result.inputArguments = inputArguments;
            result.outputArguments = outputArguments;
            result.iArguments = iArguments;
            result.tArguments = tArguments;
            result.inplaceCall = inplaceCall;
            result.hash = opHash;

            return result;
        }
    }
}
