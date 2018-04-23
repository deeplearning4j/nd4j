package org.nd4j.serde.json;

import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

@JsonDeserialize(using = LegacyIActivationDeserializer.class)
public class LegacyILossFunctionDeserializerHelper {
    private LegacyILossFunctionDeserializerHelper(){ }
}
