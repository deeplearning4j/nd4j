package org.nd4j.linalg.heartbeat;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.heartbeat.utils.EnvironmentUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * Heartbeat implementation for ND4j
 *
 * @author raver119@gmail.com
 */
public class Heartbeat {
    private static final Heartbeat INSTANCE = new Heartbeat();
    private volatile long serialVersionID;
    private AtomicBoolean enabled = new AtomicBoolean(false);
    private long cId;
    private static Logger logger = LoggerFactory.getLogger(Heartbeat.class);

    protected Heartbeat() {
        try {
            java.util.logging.Logger.getLogger("org.apache.http").setLevel(java.util.logging.Level.OFF);
            java.util.logging.Logger.getLogger("org.apache.http.wire").setLevel(java.util.logging.Level.OFF);
            java.util.logging.Logger.getLogger("org.apache.http.headers").setLevel(java.util.logging.Level.OFF);
            System.setProperty("org.apache.commons.logging.simplelog.showdatetime", "true");
            System.setProperty("org.apache.commons.logging.simplelog.log.httpclient.wire", "ERROR");
            System.setProperty("org.apache.commons.logging.simplelog.log.org.apache.http", "ERROR");
            System.setProperty("org.apache.commons.logging.simplelog.log.org.apache.http.headers", "ERROR");

            ch.qos.logback.classic.Logger LOG = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger("org.apache.http");
            LOG.setLevel(Level.OFF);
        } catch (Exception e) {

        }

        try {
            cId = EnvironmentUtils.buildCId();
        } catch (Exception e) {
            ;
        }
    }

    public static Heartbeat getInstance() {
        return INSTANCE;
    }

    public void disableHeartbeat() {
        this.enabled.set(false);
    }

    public synchronized void reportEvent(Event event, Environment environment, Task task) {
    }

    public synchronized void derivedId(long id) {
        if (serialVersionID != 0) {
            serialVersionID = id;
        }
    }

}
