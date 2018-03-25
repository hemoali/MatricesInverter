#include <iostream>
#include "xtensor/xarray.hpp"
#include "MatricesInverter.h"

using namespace std;
using namespace xt;

int main()
{
    xarray<double> testcase
            {
                    {
                            {2, -1,  0,    0, 0},
                            {-1, 2, -1,  0, 0},
                            {0, -1,  2,   -1,   0},
                            {0, 0,   -1, 2, -1},
                            {0,  0,   0, -1,   2}
                    },
                    {
                            {1, 0.1, -0.2, 3, 0},
                            {0,  2, 0.1, 2, 0.1},
                            {5, 0.9, 0.2, -0.3, 3},
                            {1, 0.3, 3,  7, 0.1},
                            {10, 0.3, 5, 0.69, 1}
                    },
            };

    MatricesInverter matricesInverter;

    cout << matricesInverter.invertMatrices(testcase);

    return 0;
}
