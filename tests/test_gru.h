#ifndef __TEST_GRU_H__
#define __TEST_GRU_H__

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>

class TestGRU : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(TestGRU);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

    protected:
        void test();
};

#endif

