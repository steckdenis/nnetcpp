#ifndef __TEST_PERCEPTRON_H__
#define __TEST_PERCEPTRON_H__

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>

class TestPerceptron : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(TestPerceptron);
    CPPUNIT_TEST(testLinear);
    CPPUNIT_TEST_SUITE_END();

    protected:
        void testLinear();
};

#endif

