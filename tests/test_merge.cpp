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

#include "test_merge.h"
#include "utils.h"

#include <network.h>
#include <dense.h>
#include <mergesum.h>
#include <mergeproduct.h>

void TestMerge::testSum()
{
    // Try to approximate simple linear function
    std::vector<Vector> input;
    std::vector<Vector> output;

    for (int i=0; i<10; ++i) {
        float x = float(i) / 10;

        input.push_back(makeVector({x}));
        output.push_back(makeVector({2.0f * x + 1}));
    }

    // Network that branches and then merges
    Network *net = new Network(1);
    Dense *dense1 = new Dense(1, 0.05);
    Dense *dense2 = new Dense(1, 0.05);
    MergeSum *sum = new MergeSum;

    dense1->setInput(net->inputPort());
    dense2->setInput(net->inputPort());
    sum->addInput(dense1->output());
    sum->addInput(dense2->output());

    net->addNode(dense1);
    net->addNode(dense2);
    net->addNode(sum);

    CPPUNIT_ASSERT_MESSAGE(
        "Learning a linear function using MergeSum",
        checkLearning(net, input, output, 0.001, 1000)
    );

    delete net;
}

void TestMerge::testProduct()
{
    // Try to approximate a quadratic function, that can be expressed as
    // (ax + b)*(cx + d) by the network
    std::vector<Vector> input;
    std::vector<Vector> output;

    for (int i=0; i<60; ++i) {
        float x = float(i) / 10 - 3.0f; // x ranges from -3 to 3

        input.push_back(makeVector({x}));
        output.push_back(makeVector({(0.55f*x + 1.21f) * (-0.9f*x + 0.3f)}));
    }

    // Network that branches and then merges
    Network *net = new Network(1);
    Dense *dense1 = new Dense(1, 0.001);
    Dense *dense2 = new Dense(1, 0.001);
    MergeProduct *product = new MergeProduct;

    dense1->setInput(net->inputPort());
    dense2->setInput(net->inputPort());
    product->addInput(dense1->output());
    product->addInput(dense2->output());

    net->addNode(dense1);
    net->addNode(dense2);
    net->addNode(product);

    CPPUNIT_ASSERT_MESSAGE(
        "Learning a quadratic function using MergeProduct",
        checkLearning(net, input, output, 0.0001, 1000)
    );

    delete net;
}
