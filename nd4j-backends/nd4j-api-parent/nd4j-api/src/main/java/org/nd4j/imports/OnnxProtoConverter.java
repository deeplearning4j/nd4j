package org.nd4j.imports;

import com.google.protobuf.TextFormat;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.io.*;

public class OnnxProtoConverter implements SameDiffProtoConverter {
    @Override
    public SameDiff importProto(File file) {
        OnnxProto3.GraphProto def = null;
        try (FileInputStream fis = new FileInputStream(file); BufferedInputStream bis = new BufferedInputStream(fis)) {
            def = OnnxProto3.GraphProto.parseFrom(bis);
        } catch (Exception e) {
            try (FileInputStream fis2 = new FileInputStream(file); BufferedInputStream bis2 = new BufferedInputStream(fis2); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
                OnnxProto3.GraphProto.Builder builder = OnnxProto3.GraphProto.newBuilder();

                StringBuilder str = new StringBuilder();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    str.append(line);//.append("\n");
                }

                TextFormat.getParser().merge(str.toString(), builder);
                def = builder.build();
            } catch (Exception e2) {
                e2.printStackTrace();
            }
        }

        if (def == null)
            throw new ND4JIllegalStateException("Unknown format: " + file.getAbsolutePath());


        return null;
    }
}
