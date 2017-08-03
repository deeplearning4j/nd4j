package org.nd4j.linalg.versioncheck;

import lombok.extern.slf4j.Slf4j;

import java.util.*;

/**
 * Utility to check versions of dependencies (DL4J, ND4J etc) to detect and warn
 * that (likely) incompatible versions are present on the classpath
 *
 * @author Alex Black
 */
@Slf4j
public class VersionCheck {

    public static final String SUPPRESS_VERSION_WARNING = "org.nd4j.versioncheck.suppress";

    private VersionCheck() {

    }

    public static void checkVersions(){

        boolean ignore = Boolean.parseBoolean(System.getProperty(SUPPRESS_VERSION_WARNING));
        if(ignore){
            return;
        }


        List<VersionInfo> l = getVersionInfo();

        Set<String> versions = new HashSet<>();

        for(VersionInfo vi : l){
            String version = vi.getVersion();
            if(version != null){
                versions.add(version);
            }
        }

        if(versions.size() > 1){
            log.warn("ND4J FAILED VERSION CHECK - {} DIFFERENT VERSIONS FOUND", versions.size());
            logVersions();
        }

        List<String> legacy = legacyVersionsPresent();
        if( legacy.size() > 0){
            log.warn("ND4J FAILED VERSION CHECK - PRE-0.9.1 VERSIONS FOUND");
            for(String s : legacy){
                log.info("Old version: {}", s);
            }
        }
    }

    private static List<String> legacyVersionsPresent(){

        //Check for 0.9.0 or earlier of: DataVec, nd4j-native, nd4j-cuda, DL4J
        //We know if the specified classes exist, and the version info class doesn't, it must be
        // a 0.9.0 or earlier version
        //This isn't a particularly elegant solution, but it should work


        String datavecClass = "org.datavec.api.writable.DoubleWritable";    //Arbitrary class that should have been stable
        String datavecVersionInfoClass = "org.datavec.api.versioncheck.DataVecApiVersionInfo";
        boolean oldDatavecVersion = checkOldVersion(datavecClass, datavecVersionInfoClass);

        String dl4jClass = "org.deeplearning4j.nn.conf.MultiLayerConfiguration";
        String dl4jVersionInfoClass = "org.deeplearning4j.versioninfo.Dl4jNnVersionInfo";
        boolean oldDl4jVersion = checkOldVersion(dl4jClass, dl4jVersionInfoClass);

        //TODO: Arbiter, RL4J

        if(!oldDatavecVersion && !oldDl4jVersion){
            return Collections.emptyList();
        }
        List<String> ret = new ArrayList<>(2);
        if(oldDatavecVersion){
            ret.add("org.datavec : datavec-api : (unknown pre-0.9.1 version)");
        }
        if(oldDl4jVersion){
            ret.add("org.deeplearning4j : deeplearning4j-nn : (unknown pre-0.9.1 version)");
        }

        return ret;
    }

    private static boolean checkOldVersion(String someClass, String versionInfoClass ){
        try{
            Class.forName(someClass);
            //No exception: does exist -> dependency is on classpath
            try{
                Class.forName(versionInfoClass);
            } catch (ClassNotFoundException e){
                //Version info class not found -> dependency IS on classpath, but it's a 0.9.0 or earlier version
                return true;
            }
        } catch (ClassNotFoundException e){
            //Not found -> not on classpath
        }
        return false;
    }

    public static void logVersions(){
        List<VersionInfo> l = getVersionInfo();

        log.info("Found {} artifacts registered for ND4J version check", l.size());

        for(VersionInfo vi : l ){
            log.info(vi.getGAV());
        }
        for(String s : legacyVersionsPresent()){
            log.info(s);
        }
    }

    public static List<VersionInfo> getVersionInfo(){
        ServiceLoader<VersionInfo> sl = ServiceLoader.load(VersionInfo.class);
        Iterator<VersionInfo> iter = sl.iterator();
        List<VersionInfo> out = new ArrayList<>();
        while(iter.hasNext()){
            out.add(iter.next());
        }

        return out;
    }
}
