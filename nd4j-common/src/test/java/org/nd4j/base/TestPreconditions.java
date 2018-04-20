package org.nd4j.base;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public class TestPreconditions {

    @Test
    public void testPreconditions(){

        Preconditions.checkArgument(true);
        try{
            Preconditions.checkArgument(false);
        } catch (IllegalArgumentException e){
            assertNull(e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here", 10);
        try{
            Preconditions.checkArgument(false, "Message %s here", 10);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here %s there", 10, 20);
        try{
            Preconditions.checkArgument(false, "Message %s here %s there", 10, 20);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here 20 there", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here %s there %s more", 10, 20, 30);
        try{
            Preconditions.checkArgument(false, "Message %s here %s there %s more", 10, 20, 30);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here 20 there 30 more", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here", 10L);
        try{
            Preconditions.checkArgument(false, "Message %s here", 10L);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here %s there", 10L, 20L);
        try{
            Preconditions.checkArgument(false, "Message %s here %s there", 10L, 20L);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here 20 there", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here %s there %s more", 10L, 20L, 30L);
        try{
            Preconditions.checkArgument(false, "Message %s here %s there %s more", 10L, 20L, 30L);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here 20 there 30 more", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here %s there %s more", "A", "B", "C");
        try{
            Preconditions.checkArgument(false, "Message %s here %s there %s more", "A", "B", "C");
        } catch (IllegalArgumentException e){
            assertEquals("Message A here B there C more", e.getMessage());
        }


    }

    @Test
    public void testPreconditionsMalformed(){

        //No %s:
        Preconditions.checkArgument(true, "This is malformed", "A", "B", "C");
        try{
            Preconditions.checkArgument(false, "This is malformed", "A", "B", "C");
        } catch (IllegalArgumentException e){
            assertEquals("This is malformed [A,B,C]", e.getMessage());
        }

        //More args than %s:
        Preconditions.checkArgument(true, "This is %s malformed", "A", "B", "C");
        try{
            Preconditions.checkArgument(false, "This is %s malformed", "A", "B", "C");
        } catch (IllegalArgumentException e){
            assertEquals("This is A malformed [B,C]", e.getMessage());
        }

        //No args
        Preconditions.checkArgument(true, "This is %s %s malformed");
        try{
            Preconditions.checkArgument(false, "This is %s %s malformed");
        } catch (IllegalArgumentException e){
            assertEquals("This is %s %s malformed", e.getMessage());
        }

        //More %s than args
        Preconditions.checkArgument(true, "This is %s %s malformed", "A");
        try{
            Preconditions.checkArgument(false, "This is %s %s malformed", "A");
        } catch (IllegalArgumentException e){
            assertEquals("This is A %s malformed", e.getMessage());
        }




    }



}
