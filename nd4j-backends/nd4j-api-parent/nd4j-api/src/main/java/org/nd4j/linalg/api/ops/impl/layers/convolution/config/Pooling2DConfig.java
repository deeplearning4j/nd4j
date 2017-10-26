package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;

@Builder
@AllArgsConstructor
@Data
public class Pooling2DConfig {

    private int kh, kw, sy, sx, ph, pw, dh, dw,virtualHeight,virtualWidth;
    private double extra;
    private Pooling2D.Pooling2DType type;
    private boolean isSameMode;


}
