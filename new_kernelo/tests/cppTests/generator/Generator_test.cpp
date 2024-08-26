#include <gtest/gtest.h>
#include "../../../src/generator/Generator.hpp"
#include "../../../src/generator/RandomGenerator.hpp"
#include "../../../src/generator/SobolGenerator.hpp"

using namespace DataGeneration;

class GeneratorTest : public testing::TestWithParam<std::string>
{
protected:
    std::shared_ptr<Generator> generator;
    unsigned int seed = 12345;
    unsigned int N = 5;
    unsigned int L = 5;
};

TEST_F(GeneratorTest, RandomGenerator5x5Seed12345)
{
    mat X_gen = mat(N, L);
    mat X_gen_seed_12345 = mat(N, L);
    X_gen_seed_12345 = {
        {0.3576, 0.2077, 0.0039, 0.4106, 0.0039},
        {0.4004, 0.0287, 0.0130, 0.2608, 0.9401},
        {0.6894, 0.6889, 0.4204, 0.0267, 0.6913},
        {0.5597, 0.4693, 0.6162, 0.1079, 0.9187},
        {0.5745, 0.2072, 0.8949, 0.3667, 0.4197}};
    generator = std::shared_ptr<Generator>(new RandomGenerator(seed));
    generator->execute(X_gen);
    ASSERT_TRUE(approx_equal(X_gen, X_gen_seed_12345, "absdiff", 1e-4));
}

TEST_F(GeneratorTest, SobolGenerator5x5)
{
    mat X_gen = mat(N, L);
    mat X_gen_sobol = mat(N, L);
    X_gen_sobol = {
        {0.5000,   0.5000,   0.5000,   0.5000,   0.5000},
        {0.7500,   0.2500,   0.2500,   0.2500,   0.7500},
        {0.2500,   0.7500,   0.7500,   0.7500,   0.2500},
        {0.3750,   0.3750,   0.6250,   0.8750,   0.3750},
        {0.8750,   0.8750,   0.1250,   0.3750,   0.8750}};
    generator = std::shared_ptr<Generator>(new SobolGenerator());
    generator->execute(X_gen);
    ASSERT_TRUE(approx_equal(X_gen, X_gen_sobol, "absdiff", 1e-4));
}
