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

/**
 * JMX MBean for managing log4j level.
 * 
 *
 */
public interface LoggingConfigureMBean {

    /**
     * Get all loggers and their level.
     *
     * @return all available loggers
     */
    public String[] getLoggers();

    /**
     * Assigns the given level to the given class.
     * 
     * @param target the FQCN of the class
     * @param levelString
     * @return
     */
    public String assignLogLevel(String target, String levelString);

    /**
     * reloads the log4j configuration from the <code>log4j.properties</code> file in the classpath.
     */
    public void resetConfiguration();
}
