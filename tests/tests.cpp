#include "test_perceptron.h"

#include <iostream>

#include <cppunit/TestRunner.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>

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

    if(!has_suite)
    {
        std::cout << "test case " << test_name << " does not exist" << std::endl;
        return EXIT_FAILURE;
    }

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