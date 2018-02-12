package org.nd4j.imports.descriptors.properties.adapters;

import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.tensorflow.framework.DataType;

import java.lang.reflect.Field;

public class DataTypeAdapter implements AttributeAdapter {

    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        on.setValueFor(fieldFor,dtypeConv((DataType) inputAttributeValue));
    }

    protected DataBuffer.Type dtypeConv(DataType dataType) {
        val x = dataType.getNumber();

        return null;
    };
}
