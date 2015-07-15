#include "test_perceptron.h"
#include "utils.h"

#include <network.h>
#include <dense.h>
#include <activation.h>

void TestPerceptron::testLinear()
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
    Dense *dense1 = new Dense(1, 0.05);

    dense1->setInput(net->inputPort());
    net->addNode(dense1);

    CPPUNIT_ASSERT_MESSAGE(
        "Learning a linear function using no hidden layer",
        checkLearning(net, input, output, 0.0001, 100)
    );

    delete net;

    // Network a an hidden layer of 10 neurons
    Dense *dense2;

    net = new Network(1);
    dense1 = new Dense(10, 0.01);
    dense2 = new Dense(1, 0.01);

    dense1->setInput(net->inputPort());
    dense2->setInput(dense1->output());

    net->addNode(dense1);
    net->addNode(dense2);

    CPPUNIT_ASSERT_MESSAGE(
        "Learning a linear function using one hidden layer",
        checkLearning(net, input, output, 0.0001, 100)
    );

    delete net;
}

void TestPerceptron::testTanh()
{
    // Try to approximate a cosinus function
    std::vector<Vector> input;
    std::vector<Vector> output;

    for (int i=0; i<300; ++i) {
        float x = float(i) / 100.0f - 1.0f;

        input.push_back(makeVector({x}));
        output.push_back(makeVector({std::sin(x)}));
    }

    // Network with a single hidden layer (with tanh activation), 10 hidden neurons
    Network *net = new Network(1);
    Dense *dense1 = new Dense(10, 0.001);
    TanhActivation *tanh1 = new TanhActivation;
    Dense *dense2 = new Dense(1, 0.001);
    TanhActivation *tanh2 = new TanhActivation;

    dense1->setInput(net->inputPort());
    tanh1->setInput(dense1->output());
    dense2->setInput(tanh1->output());
    tanh2->setInput(dense2->output());

    net->addNode(dense1);
    net->addNode(tanh1);
    net->addNode(dense2);
    net->addNode(tanh2);

    CPPUNIT_ASSERT_MESSAGE(
        "Learning a cosinus function using 10 hidden neurons",
        checkLearning(net, input, output, 0.0005, 1000)
    );

    delete net;
}
