package org.nd4j.linalg.api.environment;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * An environment descriptor
 * representing the state of
 * the system nd4j is running.
 * The fields here include:
 * cpu ram
 * number of cpu cores
 * number of gpus
 * the amount of total ram for each gpu this backend is using (indexed by device ordering, you can usually see this from nvidia-smi)
 * the blas vendor (typically openblas or cublas)
 * the number of max threads for blas
 * the number of open mp threads being used
 *
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Nd4jEnvironment implements Serializable {
    private long ram;
    private int numCores;
    private String os;
    private int numGpus;
    private List<Long> gpuRam;
    private String blasVendor;
    private long blasThreads;
    private int ompThreads;


    public final static String CUDA_DEVICE_NAME_KEY = "cuda.deviceName";
    public final static String CUDA_FREE_MEMORY_KEY = "cuda.freeMemory";
    public final static String CUDA_TOTAL_MEMORY_KEY = "cuda.totalMemory";
    public final static String CUDA_DEVICE_MAJOR_VERSION_KEY = "cuda.deviceMajor";
    public final static String CUDA_DEVICE_MINOR_VERSION_KEY = "cuda.deviceMinor";

    public final static String BACKEND_KEY = "backend";
    public final static String CUDA_NUM_GPUS_KEY = "cuda.availableDevices";
    public final static String CUDA_DEVICE_INFORMATION_KEY = "cuda.devicesInformation";
    public final static String BLAS_VENDOR_KEY = "blas.vendor";

    public final static String OS_KEY = "os";
    public final static String RAM_KEY = "memory.available";
    public final static String CPU_CORES_KEY = "cores";

    public final static String OMP_THREADS_KEY = "omp.threads";
    public final static String BLAS_THREADS_KEY = "blas.threads";

    /**
     * Load an {@link Nd4jEnvironment} from
     * the properties returned from {@link org.nd4j.linalg.api.ops.executioner.OpExecutioner#getEnvironmentInformation()}
     * derived from {@link Nd4j#getExecutioner()}
     * @return the environment representing the system the nd4j
     * backend is running on.
     */
    public static Nd4jEnvironment getEnvironment() {
        Properties envInfo = Nd4j.getExecutioner().getEnvironmentInformation();
        Nd4jEnvironment ret = Nd4jEnvironment.builder()
                        .numCores(Integer.parseInt(envInfo.getProperty(CPU_CORES_KEY, "0")))
                        .ram(Long.parseLong(envInfo.getProperty(RAM_KEY, "0"))).os(envInfo.getProperty(OS_KEY))
                        .blasVendor(envInfo.getProperty(BLAS_VENDOR_KEY))
                        .blasThreads(Long.parseLong(envInfo.getProperty(BLAS_THREADS_KEY, "0")))
                        .ompThreads(Integer.parseInt(envInfo.getProperty(OMP_THREADS_KEY, "0")))
                        .numGpus(Integer.parseInt(envInfo.getProperty(CUDA_NUM_GPUS_KEY, "0"))).build();
        if (envInfo.containsKey(CUDA_DEVICE_INFORMATION_KEY)) {
            List<Map<String, Object>> deviceInfo = (List<Map<String, Object>>) envInfo.get(CUDA_DEVICE_INFORMATION_KEY);
            List<Long> gpuRam = new ArrayList<>();
            for (Map<String, Object> info : deviceInfo) {
                gpuRam.add(Long.parseLong(info.get(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY).toString()));
            }

            ret.setGpuRam(gpuRam);
        }


        return ret;

    }


}
