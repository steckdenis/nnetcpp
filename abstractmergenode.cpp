#include "abstractmergenode.h"

AbstractMergeNode::AbstractMergeNode()
{
}

AbstractNode::Port *AbstractMergeNode::output()
{
    return &_output;
}

void AbstractMergeNode::addInput(Port *input)
{
    unsigned int dim = input->value.rows();

    if (_inputs.size() == 0) {
        // First input to be added, its size defines the shape of the output
        _output.value = Vector::Zero(dim);
        _output.error = Vector::Zero(dim);
    }

    _inputs.push_back(input);
}

void AbstractMergeNode::clearError()
{
    _output.error.setZero();
}

void AbstractMergeNode::update()
{
}
