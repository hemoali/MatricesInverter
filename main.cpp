#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xeval.hpp"

using namespace std;
using namespace xt;

pair<xarray<double>, xarray<double>> matrixLUDecomposition(const xarray<double> &);

xarray<double> invertMatrix(xarray<double>);

bool isMatrixInvertable(const xarray<double> &);

int main()
{
    xarray<double> arr1
            {{1.0, 6.0,  11.0, 16.0, 21.0},
             {2.0, 7.0,  12.0, 17.0, 22.0},
             {3.0, 8.0,  13.0, 18.0, 23.0},
             {4.0, 9.0,  14.0, 19.0, 24.0},
             {5.0, 10.0, 15.0, 20.0, 25.0}};

    xarray<double> arr2
            {{2,  -1, 0,  0,  0},
             {-1, 2,  -1, 0,  0},
             {0,  -1, 2,  -1, 0},
             {0,  0,  -1, 2,  -1},
             {0,  0,  0,  -1, 2}};

    xarray<double> arr3
            {
                    {1,  0.1, -0.2, 3,    0},
                    {0,  2,   0.1,  2,    0.1},
                    {5,  0.9, 0.2,  -0.3, 3},
                    {1,  0.3, 3,    7,    0.1},
                    {10, 0.3, 5,    0.69, 1}
            };

    xarray<double> arr4
            {{3,   -.1, -.2},
             {.1,  7,   -.3},
             {0.3, -.2, 10}};

    matrixLUDecomposition(arr3);

    return 0;
}

pair<xarray<double>, xarray<double>> matrixLUDecomposition(const xarray<double> &arr)
{
    size_t dim = arr.shape()[0];

    xarray<double, layout_type::row_major> L(arr.shape());
    xarray<double, layout_type::row_major> U = eye<double>(dim);
    xarray<double, layout_type::row_major> I = eye<double>(dim);

    // {7.85,-19.3,71.4};//

    // Assign 1st col from arr to L
    auto &&v = view(L, all(), 0);
    v = view(arr, all(), 0);

    // Assign [(1st row)/l11] from arr to U
    auto &&b = view(U, 0, range(1, dim));
    b = view(arr, 0, range(1, dim)) / L(0);

    // Filling loop
    for (int i = 1; i < dim; ++i)
    {
        // Fill L
        xarray<double> preC = zeros<double>({dim - i});

        for (int j = i; j > 0; --j)
        {
            preC += view(L, range(i, dim), j - 1) * U(j - 1, i);
        }

        auto aC = view(arr, range(i, dim), i); // arr col#i
        auto &&LCol = view(L, range(i, dim), i);

        LCol = (aC - preC);

        // Fill U
        if (dim - i - 1 <= 0) continue;

        xarray<double> preR = zeros<double>({dim - i - 1});

        for (int j = i; j > 0; --j)
        {
            preR += view(U, j - 1, range(i + 1, dim)) * L(i, j - 1);
        }

        auto aR = view(arr, i, range(i + 1, dim)); // arr row#i
        auto &&URow = view(U, i, range(i + 1, dim));

        URow = (aR - preR) / L(i, i);
    }


    // Calculate D
    xarray<double> D = zeros<double>({dim, dim});

    auto &&DRow = view(D, 0, all());
    DRow = view(I, 0, all()) / L(0, 0);

    // Calculate D array row by row
    for (int i = 1; i < dim; ++i)
    {
        xarray<double> val = zeros<double>({dim});

        for (int j = i; j > 0; --j)
        {
            val += L(i, j - 1) * view(D, j - 1, all());
        }

        auto &&row = view(D, i, all());
        row = (view(I, i, all()) - val) / L(i, i);
    }

    // Calculate X
    xarray<double> X = zeros<double>({dim, dim});

    auto &&XRow = view(X, dim - 1, all());
    XRow = view(D, dim - 1, all());

    // Calculate D array row by row
    for (int i = static_cast<int>(dim - 2); i >= 0; --i)
    {
        xarray<double> val = zeros<double>({dim});

        for (int j = i; j < dim; ++j)
        {
            val += U(i, j + 1) * view(X, j + 1, all());
        }

        auto &&row = view(X, i, all());
        row = view(D, i, all()) - val;
    }

    cout << "Matrix" << endl << arr << endl << "Inverse" << endl << X;

    return make_pair(L, U);
}

bool isMatrixInvertable(const xarray<double> &arr)
{
    if (arr.shape()[0] != arr.shape()[1]) return false;

    return true;
}

xarray<double> invertMatrix(xarray<double> arr)
{
    return flip(arr, 0);
}
