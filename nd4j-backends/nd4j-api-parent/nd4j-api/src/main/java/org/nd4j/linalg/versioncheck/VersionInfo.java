package org.nd4j.linalg.versioncheck;

/**
 * Created by Alex on 03/08/2017.
 */
public abstract class VersionInfo {

    public String getArtifactId(){
        return this.getClass().getPackage().getImplementationTitle();
    }

    public abstract String getGroupId();

    public String getVersion(){
        return this.getClass().getPackage().getImplementationVersion();
    }

    public String getGAV(){
        return getGroupId() + ":" + getArtifactId() + ":" + getVersion();
    }

}
