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

#include "test_recurrent.h"
#include "utils.h"

#include <iostream>
#include <stdlib.h>

void TestRecurrent::testCWRNN()
{
    // Network with N CWRNN nodes of M neurons each
    static const unsigned int N = 3;
    static const unsigned int M = 10;

    Network *net = makeCWRNN(N, 1, N * M, 1, 1e-2);

    // Test this network. The network is not expected to learn the parity task
    // (it is designed for sequence modelling), but mustn't diverge.
    testNetwork(net, 0.50);
}

void TestRecurrent::testGRU()
{
    // Network with N GRU cells
    static const unsigned int N = 4;

    Network *net = makeGRU(1, N, 1, 1e-2);

    // Test this network
    testNetwork(net, 0.002);
}

void TestRecurrent::testLSTM()
{
    // Network with N LSTM cells
    static const unsigned int N = 40;

    Network *net = makeLSTM(1, N, 1, 5e-3);

    // Test this network
    testNetwork(net, 0.03);
}

void TestRecurrent::testNetwork(Network *net, float target)
{
    // Training vectors for the "compute parity" task
    std::vector<std::vector<Vector>> inputs {
        makeSequence({0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}),
        makeSequence({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}),
        makeSequence({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f}),
        makeSequence({1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}),
        makeSequence({1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
        makeSequence({1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}),
        makeSequence({0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f}),
        makeSequence({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f}),
        makeSequence({1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}),
        makeSequence({0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f}),
    };
    std::vector<std::vector<Vector>> outputs {
        makeSequence({0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f}),
        makeSequence({1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f}),
        makeSequence({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}),
        makeSequence({1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
        makeSequence({1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
        makeSequence({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f}),
        makeSequence({0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}),
        makeSequence({1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f}),
        makeSequence({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f}),
        makeSequence({0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}),
    };

    // Train the network on the input sequences (all but the last one)
    for (int iteration=0; iteration<10000; ++iteration) {
        int i = rand() % (int)inputs.size();

        checkLearning(net, inputs[i], outputs[i], 0.0, 1, false, true);
    }

    // Test the network on all the input sequences (last one included)
    std::cout << "Validation" << std::endl;

    for (std::size_t i=0; i<inputs.size(); ++i) {
        // NOTE: The MSE allowed is quite big, but this is required as correct
        //       parity nevertheless has sometimes a big error, for instance if
        //       the network overshoots (predicts 1.30 instead of 1, and -0.2
        //       instead of 0).
        CPPUNIT_ASSERT_MESSAGE(
            "A test vector failed the parity test",
            checkLearning(net, inputs[i], outputs[i], target, 1, true, true)
        );
    }

    delete net;
}
