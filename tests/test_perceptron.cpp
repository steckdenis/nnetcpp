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
        checkLearning(net, input, output, 0.002, 100)
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
        checkLearning(net, input, output, 0.002, 200)
    );

    delete net;
}

void TestPerceptron::testTanh()
{
    testActivation<TanhActivation>();
}

void TestPerceptron::testSigmoid()
{
    testActivation<SigmoidActivation>();
}

template<typename T>
void TestPerceptron::testActivation()
{
    // Try to approximate a cosinus function
    std::vector<Vector> input;
    std::vector<Vector> output;

    for (int i=0; i<300; ++i) {
        float x = float(i) / 100.0f - 1.0f;

        input.push_back(makeVector({x}));
        output.push_back(makeVector({std::cos(x)}));
    }

    // Network with a single hidden layer (with tanh activation), N hidden neurons
    static const unsigned int N = 20;

    Network *net = new Network(1);
    Dense *dense1 = new Dense(N, 3e-3);
    T *act1 = new T;
    Dense *dense2 = new Dense(1, 3e-3);

    dense1->setInput(net->inputPort());
    act1->setInput(dense1->output());
    dense2->setInput(act1->output());

    net->addNode(dense1);
    net->addNode(act1);
    net->addNode(dense2);

    CPPUNIT_ASSERT_MESSAGE(
        "Learning a cosinus function using a non-linear activation function",
        checkLearning(net, input, output, 0.001, 500)
    );

    delete net;
}
