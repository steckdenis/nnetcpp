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

#ifndef __TEST_PERCEPTRON_H__
#define __TEST_PERCEPTRON_H__

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>

class TestPerceptron : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(TestPerceptron);
    CPPUNIT_TEST(testLinear);
    CPPUNIT_TEST(testTanh);
    CPPUNIT_TEST(testSigmoid);
    CPPUNIT_TEST_SUITE_END();

    protected:
        void testLinear();
        void testTanh();
        void testSigmoid();

        template<typename T>
        void testActivation();
};

#endif

