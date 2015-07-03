/*
 * Copyright 2014 Yahoo! Inc. Licensed under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
 * agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

package org.nd4j.paramserver.pistachio.pb.util;

import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.apache.log4j.spi.LoggerRepository;

public class LoggingConfigure implements LoggingConfigureMBean {

    @SuppressWarnings("unchecked")
    @Override
    public String[] getLoggers() {
        LoggerRepository r = LogManager.getLoggerRepository();

        Enumeration<Logger> loggerList = r.getCurrentLoggers();

        Logger logger = null;
        List<String> resultList = new ArrayList<String>();
        while (loggerList.hasMoreElements()) {
            logger = (Logger) loggerList.nextElement();
            if (logger.getName().startsWith("org.nd4j.paramserver.pistachio.pb")) {
                // find level
                Level level = logger.getLevel();
                while (level == null) {
                    Logger parent = (Logger)logger.getParent();
                    if (parent != null) {
                        level = parent.getLevel();
                    }
                    else {
                        break;
                    }
                }
                if (level == null) {
                    level = LogManager.getRootLogger().getLevel();
                }

                resultList.add(logger.getName() + "\t" + level);
            }
        }
        Collections.sort(resultList);

        return (String[]) resultList.toArray(new String[] {});
    }

    @Override
    public String assignLogLevel(String target, String levelString) {
        if (StringUtils.isNotBlank(levelString)) {
            Level level = Level.toLevel(levelString.trim().toUpperCase());
            if (level != null) {
                Logger logger = LogManager.getLogger(target.trim());
                if (logger != null) {
                    logger.setLevel(level);
                    return logger.getName() + "\t" + logger.getLevel();
                }

                return "Cannot find logger for " + target;
            }
        }

        return "Cannot find level " + levelString;
    }

    @Override
    public void resetConfiguration() {
        ClassLoader cl = getClass().getClassLoader();
        LogManager.resetConfiguration();
        URL log4jprops = cl.getResource("log4j.properties");
        if (log4jprops != null) {
            PropertyConfigurator.configure(log4jprops);
        }
    }
}
