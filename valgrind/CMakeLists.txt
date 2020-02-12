cmake_minimum_required(VERSION 3.5.1)

add_executable(valgrind_test ../main.cpp)

target_link_libraries(valgrind_test PUBLIC functionalModel)

add_test(NAME valgrind_functional_test COMMAND valgrind_test)