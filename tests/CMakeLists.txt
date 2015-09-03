include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_definitions(-Wall -Wextra)

# Sequence generation experiment
add_executable(sequencegeneration
    sequencegeneration.cpp
)
target_link_libraries(sequencegeneration
    nnetcpp
)

# unit tests
add_executable(tests
    tests.cpp
    test_perceptron.cpp
    test_merge.cpp
    test_recurrent.cpp
)
target_link_libraries(tests
    ${CPPUNIT_LIBRARIES}
    nnetcpp
)

add_test(perceptron tests perceptron)
add_test(merge tests merge)
add_test(gru tests gru)
