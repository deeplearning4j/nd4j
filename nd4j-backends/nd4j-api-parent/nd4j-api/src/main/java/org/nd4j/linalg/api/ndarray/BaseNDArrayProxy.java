/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 *
 */

package org.nd4j.linalg.api.ndarray;


import com.google.common.primitives.Ints;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.instrumentation.Instrumentation;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.iter.FirstAxisIterator;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.Min;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarNotEquals;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.SubOp;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.nd4j.linalg.util.NDArrayMath;
import org.nd4j.linalg.api.shape.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.lang.Iterable;
import java.nio.IntBuffer;
import java.util.*;
import java.util.Set;

import static org.nd4j.linalg.factory.Nd4j.createUninitialized;

/**
 *
 * @author Susan Eraly
 */
public class BaseNDArrayProxy implements java.io.Serializable {

    /**
     *
     */

    protected int[] arrayShape;
    protected int length;
	protected char arrayOrdering;
    protected transient  DataBuffer data;    

    public BaseNDArrayProxy(INDArray anInstance) {
		this.arrayShape = anInstance.shape();
		this.length = anInstance.length();
		this.arrayOrdering = anInstance.ordering();
		this.data =  anInstance.data();
        if(anInstance.isView()){
            this.data = anInstance.dup().data();
		}
    }
	// READ DONE HERE
	private Object readResolve() throws java.io.ObjectStreamException {
        return Nd4j.create(this.data,this.arrayShape,this.arrayOrdering);
    }

    private void readObject(ObjectInputStream s) throws IOException, ClassNotFoundException {
        try {
			//Should have array shape and ordering here
            s.defaultReadObject();
			//Need to call deser explicitly on data buffer
            read(s);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }
    //Custom deserialization for Java serialization
    protected void read(ObjectInputStream s) throws IOException, ClassNotFoundException {
		System.out.println("SUSAN_INFO: In read, PRINT CUSTOM SER/DE-SER");
        data = Nd4j.createBuffer(length,false);
        data.read(s);
    }

	// WRITE DONE HERE
    private void writeObject(ObjectOutputStream out) throws IOException {
		//takes care of everything but data buffer
        out.defaultWriteObject();
        write(out);
    }

    //Custom serialization for Java serialization
    protected void write(ObjectOutputStream out) throws IOException {
		System.out.println("SUSAN_INFO: In write, PRINT CUSTOM SER/DE-SER");
        data.write(out);
    }

}
