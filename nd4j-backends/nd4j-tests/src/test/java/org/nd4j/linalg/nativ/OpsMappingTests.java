package org.nd4j.linalg.nativ;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.reflections.Reflections;
import org.reflections.scanners.SubTypesScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertTrue;

/**
 * This unit contains tests for c++ --- java ops mapping
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class OpsMappingTests {

    @Test
    public void testCustomOpsMapping() {
        Nd4j.create(1);
    }

    @Test
    public void testLegacyOpsMapping() {
        Nd4j.create(1);

        val str = NativeOpsHolder.getInstance().getDeviceNativeOps().getAllOperations().replaceAll("simdOps::","").replaceAll("randomOps::","");

        val missing = new ArrayList<String>();

        //parsing individual groups first

        val groups = str.split(">>");
        for (val group: groups) {
            val line = group.split(" ");
            val bt = Integer.valueOf(line[0]).byteValue();
            val ops = line[1].split("<<");

            val list = getOperations(SameDiff.getTypeFromByte(bt));

            for (val op: ops) {
                val args = op.split(":");
                val hash = Long.valueOf(args[0]).longValue();
                val opNum = Long.valueOf(args[1]).longValue();
                val name = args[2];

                //log.info("group: {}; hash: {}; name: {};", SameDiff.getTypeFromByte(bt), hash, name);
            }
        }

        //log.info("Ops: {}", str);

        if (missing.size() > 0) {

            log.info("{} ops missing!", missing.size());
            log.info("{}", missing);
            assertTrue(false);
        }
    }


    protected void addOperation(Class<? extends DifferentialFunction> clazz, List<Operation> list) {
        if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface())
            return;

        try {
            DifferentialFunction node = clazz.newInstance();
            list.add(new Operation(Long.valueOf(node.opNum()), node.opName()));
        } catch (UnsupportedOperationException e) {
            //
        } catch (Exception e) {
            log.info("Failed on [{}]", clazz.getSimpleName());
            throw new RuntimeException(e);
        }
    }

    protected List<Operation> getOperations(@NonNull Op.Type type) {
        val list = new ArrayList<Operation>();

        Reflections f = new Reflections(new ConfigurationBuilder()
                .filterInputsBy(new FilterBuilder().include(FilterBuilder.prefix("org.nd4j.*")).exclude("^(?!.*\\.class$).*$"))
                .setUrls(ClasspathHelper.forPackage("org.nd4j")).setScanners(new SubTypesScanner()));


        switch (type) {
            case INDEXREDUCE: {
                Set<Class<? extends BaseIndexAccumulation>> clazzes = f.getSubTypesOf(BaseIndexAccumulation.class);

                for (Class<? extends DifferentialFunction> clazz : clazzes)
                    addOperation(clazz, list);
            }
            break;
            case REDUCE3:
            case REDUCE: {
                Set<Class<? extends BaseAccumulation>> clazzes = f.getSubTypesOf(BaseAccumulation.class);

                for (Class<? extends DifferentialFunction> clazz : clazzes)
                    addOperation(clazz, list);
            }
            break;
            case BROADCAST: {
                Set<Class<? extends BaseBroadcastOp>> clazzes = f.getSubTypesOf(BaseBroadcastOp.class);

                for (Class<? extends DifferentialFunction> clazz : clazzes)
                    addOperation(clazz, list);
            }
            break;
            case SCALAR: {
                Set<Class<? extends BaseScalarOp>> clazzes = f.getSubTypesOf(BaseScalarOp.class);

                for (Class<? extends DifferentialFunction> clazz : clazzes)
                    addOperation(clazz, list);
            }
            break;
            case PAIRWISE:
            case TRANSFORM: {
                Set<Class<? extends BaseTransformOp>> clazzes = f.getSubTypesOf(BaseTransformOp.class);

                for (Class<? extends DifferentialFunction> clazz : clazzes)
                    addOperation(clazz, list);
            }
            break;
            case CUSTOM: {
                Set<Class<? extends DynamicCustomOp>> clazzes = f.getSubTypesOf(DynamicCustomOp.class);

                for (Class<? extends DifferentialFunction> clazz : clazzes) {
                    if (clazz.getSimpleName().equalsIgnoreCase("dynamiccustomop"))
                        continue;

                    addOperation(clazz, list);
                }
            }
            break;
            //default:
            //    throw new ND4JIllegalStateException("Unknown operation type: " + type);
        }


        log.info("Group: {}; List size: {}", type, list.size());

        return list;
    }


    protected static class Operation extends Pair<Long, String> {
        protected Operation(Long opNum, String name) {
            super(opNum, name);
        }
    }
}
