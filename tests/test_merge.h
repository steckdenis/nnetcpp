#ifndef __TEST_MERGE_H__
#define __TEST_MERGE_H__

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>

class TestMerge : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(TestMerge);
    CPPUNIT_TEST(testSum);
    CPPUNIT_TEST(testProduct);
    CPPUNIT_TEST_SUITE_END();

    protected:
        void testSum();
        void testProduct();
};

#endif

