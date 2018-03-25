//
// Created by Ibrahim Radwan on 3/25/18.
//

#include "MatricesInverter.h"

/**
 * Calculate matrices inverses
 *
 *
 *
 * Assumptions:
 * [1] The matrices are squares
 * [2] The given matrices are invertible (the non invertible ones
 *      will yeild a nan. matrix)
 *
 *
 * The general idea is to follow the following steps:
 * [1] Decompose the given each matrix into its LU matrices (using Crout Decomposition)
 * [2] Calculate the intermediate solution
 * [3] Calculate the final solution (matrix inverse)
 *
 *
 *
 * The steps are being taken from this online article with some modifications:
 * https://www.gamedev.net/articles/programming/math-and-physics/matrix-inversion-using-lu-decomposition-r3637/
 *
 *
 *
 * @param matrices This array of shape (#matrices, dim, dim), where dim is the matrix length/width
 * @return xarray of shape (#matrices, dim, dim) contains the inverses
 */
xarray<double>
MatricesInverter::invertMatrices(const xarray<double> &matrices)
{
    // Get matrices count (a.k.a depth), and each matrix dimension
    size_t depth = matrices.shape()[0];
    size_t dim = matrices.shape()[1];

    // Decompose matrices to LU matrices
    pair<xarray<double>, xarray<double>> &&LU = matricesLUDecomposition(matrices, depth, dim);

    // Calculate intermediate solutions
    xarray<double> &&D = calculateIntermediateSolution(LU.first, LU.second, depth, dim);

    // Return the final matrices
    return calculateFinalSolution(LU.first, LU.second, D, depth, dim);
}

/**
 * Decompose the given matrices using Crout decomposition
 *
 *
 * This function follows these steps:
 * [A0] The first L-col and U-row don't need some operations, so we skip those operation in case of i == 0
 * [A1] Fill the L, U matrices:
 *      [A11] First fill the L-col then the U-row (as U-row depends on the previous L-col)
 *
 * How L-cols are filled (check the document mentioned above for more info):
 * [B1] Multiply the previous L-cols with the corresponding U-col (Each L-col gets multiplied by a single value from the
 *      corresponding U-col)
 * [B2] Sum results from step 1
 * [B3] Subtract the result from step 2, from the corresponding original matrix column
 *
 * How U-rows are filled (check the document mentioned above for more info):
 * [C1] Multiply the previous R-rows with the corresponding L-row (Each U-row gets multiplied by a single value from the
 *      corresponding L-row)
 * [C2] Sum results from step 1
 * [C3] Subtract the result from step 2, from the corresponding original matrix column
 * [C4] Divide the result from step 3 by L(i,i) where i = index of the current U-row we fill
 *
 *
 * The formulas:
 * l(i1) = a(i1),                                       for i=1,2,⋯,n
 * u1j = a(1j)/l(11),                                   for j=2,3,⋯,n
 *
 * For j=2,3,⋯,n−1
 * l(ij) = a(ij)− ∑{k=[1, j−1]} l(ik) * u(kj),          for i=j,j+1,⋯,n
 *
 * u(jk) = a(jk) − (∑{i=[1,j−1]} l(ji)u(ik)) / l(jj),   for k=j+1,j+2,⋯,n
 *
 *
 * @param matrices Array of matrices to be decomposed
 * @param depth Count of the matrices
 * @param dim Size of the matrix
 *
 * @return pair of 2 xarrays {@first contains all matrices L triangles}, {@second contains all matrices U triangles}
 */
pair<xarray<double>, xarray<double>>
MatricesInverter::matricesLUDecomposition(const xarray<double> &matrices, const size_t depth, const size_t dim)
{
    // Each matrix will have its own Lower triangle matrix and Upper triangle matrix
    // Crout assumes initial U as an eye matrix
    xarray<double> L = zeros<double>({depth, dim, dim});
    xarray<double> U = eye<double>({depth, dim, dim});

    // In this loop we fill L, U
    // We use same loop to reduce complexity
    for (int i = 0; i < dim; ++i)
    {
        // ========================================
        // Fill L cols
        // ========================================

        // [B3] Update the L column
        auto &&LCol = view(L, all(), range(i, dim), i);
        LCol = (view(matrices, all(), range(i, dim), i) - getLColSubtractedValue(L, U, i, dim));

        // ========================================
        // Fill U rows
        // ========================================

        // U rows finish calculations earlier than L-cols by 1 iteration
        if (dim - i - 1 <= 0) continue;

        // [C4] Get the corresponding L(i,i) value
        xarray<double> LCorrespondingDivisorValue = view(L, all(), i, i);
        LCorrespondingDivisorValue.reshape({LCorrespondingDivisorValue.shape()[0], 1});

        // [C3,4]
        auto &&URow = view(U, all(), i, range(i + 1, dim));
        URow = ((view(matrices, all(), i, range(i + 1, dim)) - getURowSubtractedValue(L, U, i, dim)) /
                    LCorrespondingDivisorValue);
    }

    return make_pair(L, U);
}

/**
 * Calculate the value to be subtracted from the current L-Col
 *
 * @param L Matrices Lower triangle
 * @param R Matrices Upper triangle
 * @param LColIdx Index of the column we are calculating
 * @param dim Matrix dimension
 *
 * @return xarray the values to be subtracted from the original matrices columns
 */
xarray<double>
MatricesInverter::getLColSubtractedValue(const xarray<double> &L, const xarray<double> &U, int LColIdx, int dim)
{
    xarray<double> LColSubtractedValue; // Value to be subtracted later from the corresponding original matrix column

    // [A0] For the first column we subtract nothing
    if (!LColIdx)
    {
        LColSubtractedValue = 0;
    } else
    {
        // [B1] Calculate the subtractedValue
        xarray<double> UCol = view(U, all(), range(0, LColIdx), LColIdx);
        xarray<double> previousLCols = view(L, all(), range(LColIdx, dim), range(0, LColIdx));

        UCol.reshape({UCol.shape()[0], 1, UCol.shape()[1]});

        // [B2] Calculate the subtractedValue
        LColSubtractedValue = (sum((previousLCols * UCol), {2}));
    }

    return LColSubtractedValue;
}

/**
 * Calculate the value to be subtracted from the current L-Col
 *
 * @param L Matrices Lower triangle
 * @param R Matrices Upper triangle
 * @param URowIdx Index of the column we are calculating
 * @param dim Matrix dimension
 *
 * @return xarray the values to be subtracted from the original matrices columns
 */
xarray<double>
MatricesInverter::getURowSubtractedValue(const xarray<double> &L, const xarray<double> &U, int URowIdx, int dim)
{
    xarray<double> URowSubtractedValue; // Value to be subtracted later from the corresponding original matrix column

    // [A0] For the first row we subtract nothing
    if (!URowIdx)
    {
        URowSubtractedValue = 0;
    } else
    {
        // [C1] Calculate the subtractedValue
        xarray<double> previousURows = view(U, all(), range(0, URowIdx), range(URowIdx + 1, dim));
        xarray<double> LRow = view(L, all(), URowIdx, range(0, URowIdx));

        LRow.reshape({LRow.shape()[0], LRow.shape()[1], 1});

        // [C2]
        URowSubtractedValue = (sum((previousURows * LRow), {1}));
    }

    return URowSubtractedValue;
}

/**
 * Calculate the intermediate solutions of the system
 *
 *
 *
 * This function follows these steps:
 * [A0] First value in each depth-level doesn't require subtraction
 * [A1] Multiply previous D-values by corresponding L-row
 * [A2] Sum the values from step 1
 * [A3] Subtract the result from step 2 from the corresponding I (eye matrix) column
 * [A4] Divide the result by L(i,i) where i = idx of the current D value we calculate
 *
 *
 * The formulas:
 * d(1) = b(1) / l(11)
 * d(i) = b(i) − (∑{j=[1,i-1]} l(ij) * d(j)) / lii,     for i=2,3,⋯,n
 *
 * @param L
 * @param U
 * @param depth Count of the matrices
 * @param dim Size of the matrix
 *
 * @return xarray The intermediate solutions array
 */
xarray<double>
MatricesInverter::calculateIntermediateSolution(const xarray<double> &L, const xarray<double> &U, const size_t depth,
                                                const size_t dim)
{
    // The eye matrix
    xarray<double> I = eye<double>({depth, dim, dim});

    // Calculate D(intermediate solution) matrix
    xarray<double> D = zeros<double>({depth, dim, dim});

    for (int i = 0; i < dim; ++i)
    {
        xarray<double> DSubtractedValue;
        // [A0] first value doesn't need subtraction
        if (!i)
        {
            DSubtractedValue = 0;
        } else
        {
            // A[1]
            xarray<double> previousDRows = (view(D, all(), range(0, i), all()));
            xarray<double> LRow = (view(L, all(), i, range(0, i)));

            LRow.reshape({LRow.shape()[0], LRow.shape()[1], 1});

            // A[2]
            DSubtractedValue = (sum((previousDRows * LRow), {1}));
        }

        auto &&row = view(D, all(), i, all());

        // A[3] get the divisor value
        xarray<double> LCorrespondingDivisorValue = view(L, all(), i, i);
        LCorrespondingDivisorValue.reshape({depth, 1});

        // A[3,4]
        row = ((view(I, all(), i, all()) - DSubtractedValue) / LCorrespondingDivisorValue);
    }

    return D;
}

/**
 * Calculate the final solutions of the system
 *
 *
 *
 * This function follows these steps (we calculate from the end to the begining):
 * [A0] Final value in each depth-level doesn't require subtraction
 * [A1] Multiply next D-values by corresponding U-row
 * [A2] Sum the values from step 1
 * [A3] Subtract the result from step 2 from the corresponding D (intermediate matrix) value
 *
 *
 * The formulas:
 * x(n) = d(n)
 * x(i) = d(i) − ∑{j=[i+1,n]} u(ij)*x(j),               for i=n−1,n−2,⋯,1
 *
 * @param L Lower triangle
 * @param U Upper triangle
 * @param D Intermediate solution matrix
 * @param dim Matrix dimension
 *
 * @return xarray The intermediate solutions array
 */
xarray<double>
MatricesInverter::calculateFinalSolution(const xarray<double> &L, const xarray<double> &U,
                                         const xarray<double> &D, const size_t depth, const size_t dim)
{
    // Calculate final solution matrix
    xarray<double> finalSolution = zeros<double>({depth, dim, dim});

    // Calculate finalSolution array row by row
    for (int i = dim - 1; i >= 0; --i)
    {
        xarray<double> subtractedValue;

        // [A0]
        if (i == dim - 1)
        {
            // Last value doesn't require subtraction
            subtractedValue = 0;
        } else
        {
            // A[1]
            xarray<double> nextValues = (view(finalSolution, all(), range(i + 1, dim), all()));
            xarray<double> URow = (view(U, all(), i, range(i + 1, dim)));

            URow.reshape({URow.shape()[0], URow.shape()[1], 1});

            // A[1,2]
            subtractedValue = (sum((nextValues * URow), {1}));
        }

        // A[3]
        auto &&row = view(finalSolution, all(), i, all());
        row = view(D, all(), i, all()) - subtractedValue;
    }

    return finalSolution;
}
