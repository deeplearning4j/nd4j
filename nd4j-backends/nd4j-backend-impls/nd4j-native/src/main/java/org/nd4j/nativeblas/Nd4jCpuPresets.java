/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */
package org.nd4j.nativeblas;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author saudet
 */
@Properties(target = "org.nd4j.nativeblas.Nd4jCpu",
                value = {@Platform(include = {"NativeOps.h",
                                              "memory/Workspace.h",
                                              "indexing/NDIndex.h",
                                              "indexing/IndicesList.h",
                                              "NDArray.h",
                                              "graph/ArrayList.h",
                                              "NDArrayFactory.h",
                                              "graph/Variable.h",
                                              "graph/Stash.h",
                                              "graph/VariableSpace.h",
                                              "helpers/helper_generator.h",
                                              "graph/Block.h",
                                              "helpers/shape.h",
                                              "graph/ShapeList.h",
                                              "op_boilerplate.h",
                                              "ops/declarable/OpDescriptor.h",
                                              "ops/declarable/DeclarableOp.h",
                                              "ops/declarable/DeclarableReductionOp.h",
                                              "ops/declarable/DeclarableCustomOp.h",
                                              "ops/declarable/OpRegistrator.h",
                                              "ops/declarable/CustomOperations.h"},
                                compiler = "cpp11", library = "jnind4jcpu", link = "nd4jcpu", preload = "libnd4jcpu"),
                                @Platform(value = "linux", preload = "gomp@.1",
                                                preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/",
                                                                "/usr/lib/powerpc64-linux-gnu/",
                                                                "/usr/lib/powerpc64le-linux-gnu/"})})
public class Nd4jCpuPresets implements InfoMapper {
    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("thread_local", "ND4J_EXPORT", "INLINEDEF").cppTypes().annotations())
                        .put(new Info("NativeOps").base("org.nd4j.nativeblas.NativeOps"))
                        .put(new Info("char").valueTypes("char").pointerTypes("@Cast(\"char*\") String",
                                        "@Cast(\"char*\") BytePointer"))
                        .put(new Info("Nd4jPointer").cast().valueTypes("Pointer").pointerTypes("PointerPointer"))
                        .put(new Info("Nd4jIndex").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer",
                                        "long[]"))
                        .put(new Info("Nd4jStatus").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer",
                                        "int[]"))
                        .put(new Info("float16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer",
                                        "short[]"));

        infoMap.put(new Info("__CUDACC__").define(false))
               .put(new Info("MAX_UINT").translate(false))
               .put(new Info("std::initializer_list", "cnpy::NpyArray",
                             "nd4j::graph::FlatResult", "nd4j::graph::FlatVariable").skip())
               .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String")
                                           .pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::pair<int,int>").pointerTypes("IntIntPair").define())
               .put(new Info("nd4j::IndicesList").purify());

        String classTemplates[] = {
                "nd4j::NDArray",
                "nd4j::ArrayList",
                "nd4j::NDArrayFactory",
                "nd4j::graph::Variable",
                "nd4j::graph::Stash",
                "nd4j::graph::VariableSpace",
                "nd4j::graph::Block",
                "nd4j::ops::DeclarableOp",
                "nd4j::ops::DeclarableReductionOp",
                "nd4j::ops::DeclarableCustomOp",

                // DECLARE_REDUCTION_OP
                "nd4j::ops::testreduction",

                // DECLARE_OP
                "nd4j::ops::noop",
                "nd4j::ops::testop2i2o",
                "nd4j::ops::softmax",
                "nd4j::ops::softmax_bp",
                "nd4j::ops::biasadd",
                "nd4j::ops::floor",
                "nd4j::ops::realdiv",
                "nd4j::ops::merge",
                "nd4j::ops::broadcastgradientargs",
                "nd4j::ops::assign",
                "nd4j::ops::mergemax",
                "nd4j::ops::mergemaxindex",
                "nd4j::ops::mergeadd",
                "nd4j::ops::mergeavg",
                "nd4j::ops::identity",
                "nd4j::ops::add",
                "nd4j::ops::subtract",
                "nd4j::ops::reversesubtract",
                "nd4j::ops::multiply",
                "nd4j::ops::divide",
                "nd4j::ops::reversedivide",
                "nd4j::ops::reshapeas",
                "nd4j::ops::transpose",

                // DECLARE_DIVERGENT_OP
                "nd4j::ops::Switch",

                // DECLARE_CUSTOM_OP
                "nd4j::ops::testcustom",
                "nd4j::ops::concat",
                "nd4j::ops::matmul",
                "nd4j::ops::conv2d",
                "nd4j::ops::lrn",
                "nd4j::ops::reshape",
                "nd4j::ops::sconv2d",
                "nd4j::ops::sconv2d_bp",
                "nd4j::ops::deconv2d",
                "nd4j::ops::deconv2d_bp",
                "nd4j::ops::maxpool2d",
                "nd4j::ops::avgpool2d",
                "nd4j::ops::pnormpool2d",
                "nd4j::ops::maxpool3d_bp",
                "nd4j::ops::avgpool3d",
                "nd4j::ops::avgpool3d_bp",
                "nd4j::ops::fullconv3d",
                "nd4j::ops::fullconv3d_bp",
                "nd4j::ops::fullconv3d_grad",
                "nd4j::ops::maxpool2d_bp",
                "nd4j::ops::pooling2d",
                "nd4j::ops::avgpool2d_bp",
                "nd4j::ops::pnormpool2d_bp",

                // DECLARE_CONFIGURABLE_OP
                "nd4j::ops::tensormmul",
                "nd4j::ops::clipbyvalue",
                "nd4j::ops::scatter_update",
                "nd4j::ops::relu",
                "nd4j::ops::repeat",
                "nd4j::ops::randomuniform",
                "nd4j::ops::permute",
                "nd4j::ops::sum",
                "nd4j::ops::batchnorm",
                "nd4j::ops::batchnorm_bp",
                "nd4j::ops::conv3d",
                "nd4j::ops::conv3d_bp",
                "nd4j::ops::upsampling2d",
                "nd4j::ops::upsampling2d_bp",
                "nd4j::ops::maxpool3d",
                "nd4j::ops::ismax",

                "nd4j::ops::firas_sparse"};
        for (String t : classTemplates) {
            String s = t.substring(t.lastIndexOf(':') + 1);
            infoMap.put(new Info(t + "<float>").pointerTypes("Float" + s))
                   .put(new Info(t + "<float16>").pointerTypes("Half" + s))
                   .put(new Info(t + "<double>").pointerTypes("Double" + s));
        }
        infoMap.put(new Info("nd4j::ops::OpRegistrator::updateMSVC").skip());
    }
}
