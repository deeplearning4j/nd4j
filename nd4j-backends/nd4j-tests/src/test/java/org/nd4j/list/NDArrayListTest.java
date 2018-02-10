package org.nd4j.list;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class NDArrayListTest {

    @Test
    public void testList() {
        NDArrayList ndArrayList = new NDArrayList();
        List<Double> arrayAssertion = new ArrayList<>();
        for(int i = 0; i < 11; i++) {
            ndArrayList.add((double) i);
            arrayAssertion.add((double) i);
        }

        assertEquals(arrayAssertion.size(),arrayAssertion.size());
        assertEquals(arrayAssertion,ndArrayList);


        arrayAssertion.remove(1);
        ndArrayList.remove(1);
        assertEquals(ndArrayList,arrayAssertion);

        arrayAssertion.remove(2);
        ndArrayList.remove(2);
        assertEquals(ndArrayList,arrayAssertion);


        arrayAssertion.add(5,8.0);
        ndArrayList.add(5,8.0);
        assertEquals(arrayAssertion,ndArrayList);

        assertEquals(arrayAssertion.contains(8.0),ndArrayList.contains(8.0));
        assertEquals(arrayAssertion.indexOf(8.0),ndArrayList.indexOf(8.0));
        assertEquals(arrayAssertion.lastIndexOf(8.0),ndArrayList.lastIndexOf(8.0));

    }

}
