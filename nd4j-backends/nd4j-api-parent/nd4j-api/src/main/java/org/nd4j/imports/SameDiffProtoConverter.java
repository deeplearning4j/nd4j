package org.nd4j.imports;

import org.nd4j.autodiff.samediff.SameDiff;

import java.io.File;

public interface SameDiffProtoConverter {

    SameDiff importProto(File file);

}
