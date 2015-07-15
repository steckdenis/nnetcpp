include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_definitions(-Wall -Wextra)

# unit tests
add_executable(tests
    tests.cpp
    test_perceptron.cpp
    test_merge.cpp
    test_gru.cpp
)
target_link_libraries(tests
    ${CPPUNIT_LIBRARIES}
    nnetcpp
)

add_test(perceptron tests perceptron)
add_test(merge tests merge)
add_test(gru tests gru)
