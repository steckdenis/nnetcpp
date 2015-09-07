/*
 * Copyright (c) 2015 Vrije Universiteit Brussel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __TEST_RECURRENT_H__
#define __TEST_RECURRENT_H__

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>

class Network;

class TestRecurrent : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(TestRecurrent);
    CPPUNIT_TEST(testCWRNN);
    CPPUNIT_TEST(testGRU);
    CPPUNIT_TEST(testLSTM);
    CPPUNIT_TEST_SUITE_END();

    protected:
        void testCWRNN();
        void testGRU();
        void testLSTM();

    private:
        void testNetwork(Network *net, float target);
};

#endif

