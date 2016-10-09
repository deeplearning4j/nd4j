package org.nd4j.parameterserver.model;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

/**
 * Created by agibsonccc on 10/9/16.
 */
@Builder @Data
public class ServerTypeJson implements Serializable {
    private String type;
}
