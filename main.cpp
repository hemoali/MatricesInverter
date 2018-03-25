#include <iostream>
#include "xtensor/xarray.hpp"
#include "MatricesInverter.h"

using namespace std;
using namespace xt;

int main()
{
    xarray<double> arr1
            {
                    {{1.0, 6.0, 11.0, 16.0, 21.0},
                            {2.0, 7.0, 12.0, 17.0, 22.0},
                            {3.0, 8.0, 13.0, 18.0, 23.0},
                            {4.0, 9.0, 14.0, 19.0, 24.0},
                            {5.0, 10.0, 15.0, 20.0, 25.0}}
            };

    xarray<double> arr2
            {
                    {{2, -1, 0, 0, 0},
                            {-1, 2, -1, 0, 0},
                            {0, -1, 2, -1, 0},
                            {0, 0, -1, 2, -1},
                            {0, 0, 0, -1, 2}}
            };

    xarray<double> arr3
            {
                    {{1, 0.1, -0.2, 3, 0},
                            {0, 2, 0.1, 2, 0.1},
                            {5, 0.9, 0.2, -0.3, 3},
                            {1, 0.3, 3, 7, 0.1},
                            {10, 0.3, 5, 0.69, 1}}
            };

    xarray<double> arr4
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

    xarray<double> arr5
            {
                    {{3, -.1, -.2},
                            {.1, 7, -.3},
                            {0.3, -.2, 10}}
            };


    MatricesInverter matricesInverter;

    matricesInverter.invertMatrices(arr4);

    return 0;
}
