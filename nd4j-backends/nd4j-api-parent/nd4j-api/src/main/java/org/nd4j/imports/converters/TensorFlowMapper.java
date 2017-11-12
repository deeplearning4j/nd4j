package org.nd4j.imports.converters;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.reflections.Reflections;
import org.reflections.scanners.SubTypesScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

@Slf4j
public class TensorFlowMapper implements NodeMapper<NodeDef> {
    private static final TensorFlowMapper INSTANCE = new TensorFlowMapper();
    private Map<String, DifferentialFunction> nodeConverters = new HashMap<>();

    protected TensorFlowMapper() {

        Reflections f = new Reflections(new ConfigurationBuilder().filterInputsBy(
                new FilterBuilder().include(FilterBuilder.prefix("org.nd4j")).exclude("^(?!.*\\.class$).*$") //Consider only .class files (to avoid debug messages etc. on .dlls, etc
                        .exclude("^(?!org\\.nd4j\\.imports\\.converters\\.tf).*") //Exclude any not in the ops directory
        )

                .setUrls(ClasspathHelper.forPackage("org.nd4j")).setScanners(new SubTypesScanner()));

        Set<Class<? extends DifferentialFunction>> clazzes = f.getSubTypesOf(DifferentialFunction.class);

        for (Class<? extends DifferentialFunction> clazz : clazzes) {
            if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface())
                continue;

            try {
                DifferentialFunction node = clazz.newInstance();
                val name = node.opName();
                if (nodeConverters.containsKey(name)) {
                    throw new ND4JIllegalStateException("OpName duplicate found: " + name);
                } else {
                    log.info("Adding converter for [" + name + "]");
                    nodeConverters.put(name, node);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static TensorFlowMapper getInstance() {
        return INSTANCE;
    }

    @Override
    public TOp asIntermediate(NodeDef node, TGraph graph) {
        // first we try to use special converters
        val converter = nodeConverters.get(node.getOp().toLowerCase());
        if (converter != null)
            return converter.asIntermediateRepresentation(node, graph);
        else {
            val defaultConverter = nodeConverters.get("GenericOpConverter");

            return defaultConverter.asIntermediateRepresentation(node, graph);
        }
    }

    public Set<String> knownOps() {
        return nodeConverters.keySet();
    }
}
