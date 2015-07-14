#include "test_perceptron.h"
#include "utils.h"

#include <network.h>
#include <dense.h>
#include <tanh.h>

void TestPerceptron::testSingleLinear()
{
    // Try to approximate a linear function
    std::vector<Vector> input;
    std::vector<Vector> output;

    input.push_back(makeVector({-1.0}));
    input.push_back(makeVector({-0.6}));
    input.push_back(makeVector({-0.2}));
    input.push_back(makeVector({0.2}));
    input.push_back(makeVector({0.6}));
    input.push_back(makeVector({1.0}));

    output.push_back(makeVector({2.0}));
    output.push_back(makeVector({3.0}));
    output.push_back(makeVector({4.0}));
    output.push_back(makeVector({5.0}));
    output.push_back(makeVector({6.0}));
    output.push_back(makeVector({7.0}));

    // Network with a single Dense layer
    Network *net = new Network(1);
    Dense *dense = new Dense(1, 0.05);

    dense->setInput(net->inputPort());
    net->addNode(dense);

    CPPUNIT_ASSERT_MESSAGE(
        "Learning a linear function",
        checkLearning(net, input, output, 0.0001, 100)
    );
}
