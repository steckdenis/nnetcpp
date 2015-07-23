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

#include "test_perceptron.h"
#include "test_merge.h"
#include "test_recurrent.h"

#include <iostream>

#include <cppunit/TestRunner.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>

#include <fenv.h>

int main(int argc, char **argv)
{
    bool all = false;
    bool has_suite = false;

    if (argc < 2) {
        std::cout << "test <test>" << std::endl;
        return EXIT_FAILURE;
    }

    // Test names
    std::string test_name(argv[1]);

    all = (test_name == "all");

    #define TESTSUITE(name, string)                     \
    if (all || test_name == string) {                   \
        CPPUNIT_TEST_SUITE_REGISTRATION(name);          \
        has_suite = true;                               \
    }

    TESTSUITE(TestPerceptron, "perceptron");
    TESTSUITE(TestMerge, "merge");
    TESTSUITE(TestRecurrent, "recurrent");

    if(!has_suite)
    {
        std::cout << "test case " << test_name << " does not exist" << std::endl;
        return EXIT_FAILURE;
    }

    // Enable FPU exceptions so that NaN and infinites can be traced back
    feenableexcept(FE_INVALID);

    // Run the tests
    CppUnit::TestResult result;

    CppUnit::TestResultCollector collected_results;
    result.addListener(&collected_results);

    CppUnit::TestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    runner.run(result);

    CppUnit::CompilerOutputter outputter(&collected_results, std::cerr);
    outputter.write();

    return collected_results.wasSuccessful() ? EXIT_SUCCESS : EXIT_FAILURE;
}