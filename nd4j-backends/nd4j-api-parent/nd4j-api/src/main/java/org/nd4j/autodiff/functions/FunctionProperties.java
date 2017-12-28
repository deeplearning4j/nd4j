package org.nd4j.autodiff.functions;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import com.google.flatbuffers.FlatBufferBuilder;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.graph.FlatProperties;
import org.nd4j.graph.FlatResult;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

@Data
@Slf4j
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FunctionProperties {
    private String name;
    private List<Integer> i;
    private List<Long> l;
    private List<Double> d;
    private List<INDArray> a;


    public int asFlatProperties(FlatBufferBuilder bufferBuilder) {
        int iname = bufferBuilder.createString(name);
        int ii = FlatProperties.createIVector(bufferBuilder, Ints.toArray(i));
        int il = FlatProperties.createLVector(bufferBuilder, Longs.toArray(l));
        int id = FlatProperties.createDVector(bufferBuilder, Doubles.toArray(d));

        int arrays[] = new int[a.size()];
        int cnt = 0;
        for (val array: a) {
            int off = array.toFlatArray(bufferBuilder);
            arrays[cnt++] = off;
        }

        int ia = FlatProperties.createAVector(bufferBuilder, arrays);

        return FlatProperties.createFlatProperties(bufferBuilder, iname, ii, il, id, ia);
    }

    static FunctionProperties fromFlatProperties(FlatProperties properties) {
        return null;
    }
}
