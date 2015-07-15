#include "mergesum.h"

#include <iostream>

MergeSum::MergeSum()
{
}

void MergeSum::forward()
{
    _output.value.setZero();

    // Add the inputs together
    for (Port *input : _inputs) {
        _output.value.noalias() += input->value;
    }
}

void MergeSum::backward()
{
    // Send the output error to all the inputs. This can be justified by the fact
    // that this layer performs f(a, b) = a + b, so df/da = 1 + 0 = 1, so
    // error(a) = df/da * error(out) = 1 * error(out) = error(out)
    for (Port *input : _inputs) {
        input->error.noalias() += _output.error;
    }
}
