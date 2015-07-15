#include "mergeproduct.h"

#include <assert.h>

MergeProduct::MergeProduct()
{
}

void MergeProduct::forward()
{
    _output.value.setOnes();

    // Add the inputs together
    for (Port *input : _inputs) {
        _output.value.array() *= input->value.array();
    }
}

void MergeProduct::backward()
{
    // If f(a, b, c) = a * b * c, df/da = b * c = f(a, b, c) / a. So, divide the
    // output by one of the inputs in order to compute the gradient of the product
    // with regard to this input, and multiply the output error by this gradient
    // in order to produce the input error.
    for (Port *input : _inputs) {
        input->error.noalias() += _output.error.cwiseProduct(
            _output.value.cwiseQuotient(input->value)
        );
    }
}
