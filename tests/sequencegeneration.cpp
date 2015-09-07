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

#include "utils.h"

#include <string>
#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <fenv.h>

unsigned int bptt_variant = 1;

int main(int argc, char **argv)
{
    Network *network = nullptr;
    unsigned int hidden_neurons = 100;
    unsigned int epochs = 10000;
    float learning_rate = 1e-4;

    // Enable FPU exceptions so that NaN and infinites can be traced back
    feenableexcept(FE_INVALID);

    // State machine for parsing the arguments
    std::string method_name("standard");
    std::string network_name;

    enum {
        Network,
        Method,
        HiddenNeurons,
        LearningRate,
        Momentum,
        Epochs
    } state = Network;

    for (int i=1; i<argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--network") {
            state = Network;
        } else if (arg == "--method") {
            state = Method;
        } else if (arg == "--hidden") {
            state = HiddenNeurons;
        } else if (arg == "--rate") {
            state = LearningRate;
        } else if (arg == "--momentum") {
            state = Momentum;
        } else if (arg == "--epochs") {
            state = Epochs;
        } else {
            switch (state) {
            case Network:
                if (arg == "gru") {
                    network_name = "GRU";
                    network = makeGRU(1, hidden_neurons, 1, learning_rate);
                } else if (arg == "lstm") {
                    network_name = "LSTM";
                    network = makeLSTM(1, hidden_neurons, 1, learning_rate);
                } else if (arg == "cwrnn") {
                    network_name = "Clockwork RNN";
                    network = makeCWRNN(9, 1, hidden_neurons, 1, learning_rate);
                }
                break;

            case Method:
                if (arg == "stdbptt") {
                    method_name = "standard";
                    AbstractRecurrentNetworkNode::bptt_variant = AbstractRecurrentNetworkNode::Standard;
                } else if (arg == "expbptt") {
                    method_name = "experimental";
                    AbstractRecurrentNetworkNode::bptt_variant = AbstractRecurrentNetworkNode::Experimental;
                }
                break;

            case HiddenNeurons:
                hidden_neurons = atoi(argv[i]);
                break;

            case LearningRate:
                learning_rate = strtof(argv[i], nullptr);
                break;

            case Momentum:
                Dense::momentum = strtof(argv[i], nullptr);
                break;

            case Epochs:
                epochs = atoi(argv[i]);
                break;
            }
        }
    }

    if (network == nullptr) {
        std::cerr << "Provide gru or lstm as an argument in order to test GRU or LSTM" << std::endl;
        return 1;
    }

    // Sequence on which the network is trained
    std::vector<Vector> sequence = makeSequence({
        -0.03367, -0.05162, -0.05785, -0.05829, -0.05066, -0.05653, -0.05820,
        -0.04115, -0.02043, -0.00087, 0.00088, -0.00171, 0.00243, 0.01522, 0.03385,
        0.04990, 0.06215, 0.06547, 0.06494, 0.06632, 0.07205, 0.07646, 0.07481,
        0.06810, 0.06729, 0.07286, 0.06561, 0.04486, 0.01604, -0.00924, -0.03465,
        -0.06571, -0.08933, -0.11226, -0.13533, -0.15354, -0.15968, -0.15558,
        -0.16089, -0.16801, -0.17019, -0.18657, -0.22277, -0.25849, -0.28191,
        -0.30395, -0.33891, -0.38470, -0.41923, -0.44245, -0.47085, -0.49645,
        -0.50708, -0.50851, -0.50828, -0.49741, -0.47335, -0.44446, -0.41713,
        -0.39397, -0.37210, -0.36676, -0.38291, -0.39999, -0.40200, -0.39673,
        -0.39756, -0.37981, -0.33821, -0.28911, -0.24067, -0.20484, -0.17076,
        -0.13566, -0.10128, -0.07876, -0.06811, -0.04824, -0.04390, -0.05795,
        -0.05512, -0.02227, -0.00201, -0.03144, -0.05106, -0.04371, -0.04993,
        -0.05487, -0.04180, -0.03407, -0.03778, -0.02296, -0.01722, -0.03764,
        -0.04542, -0.06444, -0.11293, -0.15835, -0.16665, -0.14545, -0.12730,
        -0.09936, -0.06023, -0.03317, -0.01905, 0.01517, 0.05537, 0.05926,
        0.04241, 0.00933, 0.01444, 0.05665, 0.05627, 0.03955, 0.02992, 0.04654,
        0.09101, 0.11960, 0.10950, 0.08591, 0.10146, 0.09711, 0.04806, 0.01745,
        -0.00624, -0.05097, -0.09354, -0.09771, -0.09505, -0.11365, -0.13552,
        -0.15621, -0.16899, -0.17307, -0.18547, -0.21603, -0.24129, -0.25401,
        -0.27697, -0.31584, -0.36555, -0.42198, -0.44669, -0.42700, -0.42470,
        -0.43207, -0.42334, -0.41855, -0.39722, -0.35751, -0.34249, -0.36169,
        -0.36458, -0.35183, -0.35312, -0.35336, -0.34477, -0.31118, -0.26380,
        -0.23419, -0.21541, -0.18335, -0.13434, -0.07879, -0.01644, 0.03472,
        0.05102, 0.03221, 0.01661, 0.02299, 0.01835, 0.00895, 0.01040, 0.00083,
        -0.02173, -0.03451, -0.04335, -0.06306, -0.07747, -0.07921, -0.08067,
        -0.08810, -0.10064, -0.11303, -0.11399, -0.11094, -0.12068, -0.13160,
        -0.13164, -0.12990, -0.14770, -0.19423, -0.24731, -0.28740, -0.31461,
        -0.34304, -0.39350, -0.45157, -0.49765, -0.54330, -0.57736, -0.58819,
        -0.58721, -0.57550, -0.55177, -0.52019, -0.49210, -0.47801, -0.47133,
        -0.45379, -0.42883, -0.41992, -0.43058, -0.45351, -0.48282, -0.50285,
        -0.50449, -0.49629, -0.49761, -0.51473, -0.52904, -0.53260, -0.52902,
        -0.53238, -0.55256, -0.57817, -0.60971, -0.64427, -0.68303, -0.72298,
        -0.76154, -0.80728, -0.83626, -0.83992, -0.83242, -0.82862, -0.82565,
        -0.80799, -0.78567, -0.75777, -0.71782, -0.67034, -0.63977, -0.63538,
        -0.63348, -0.62516, -0.61300, -0.60485, -0.59716, -0.59254, -0.58835,
        -0.57021, -0.54061, -0.49884, -0.44910, -0.40085, -0.36362, -0.33658,
        -0.31449, -0.29045, -0.26110, -0.22470, -0.17177, -0.11554, -0.05367
    });

    // The input sequence is entirely made of ones
    std::vector<Vector> input;
    Vector one = makeVector({1.0f});

    for (unsigned int i=0; i<sequence.size(); ++i) {
        input.push_back(one);
    }

    // Training epochs
    std::ofstream training("training.dat");
    float avg_mse = 1.0f;

    for (unsigned int t=0; t<epochs; ++t) {
        float mse = trainNetwork(network, input, sequence, 1, false, true, false);

        training << t << ' ' << mse << std::endl;

        // Keep a running average of the MSE so that statistics can be printed
        avg_mse = 0.9f * avg_mse + 0.1f * mse;
    }


    std::cout << "Neural network used:     " << network_name << '\n';
    std::cout << "Neurons in hidden layer: " << hidden_neurons << '\n';
    std::cout << "BPTT method:             " << method_name << '\n';
    std::cout << "Learning rate:           " << learning_rate << '\n';
    std::cout << "Momentum:                " << Dense::momentum << '\n';
    std::cout << "MSE:                     " << avg_mse << std::endl;

    // Predict a sequence
    std::ofstream prediction("prediction.dat");

    network->reset();

    for (std::size_t t=0; t<sequence.size(); ++t) {
        network->setCurrentTimestep(t);

        // Print the original sequence and the predicted one
        prediction << sequence[t](0) << ' ' << network->predict(one)(0) << std::endl;
    }

    return 0;
}
