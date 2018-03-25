//
// Created by Ibrahim Radwan on 3/25/18.
//

#ifndef XTENSORPROJECT_MATRICESINVERTER_H
#define XTENSORPROJECT_MATRICESINVERTER_H

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xeval.hpp"

using namespace xt;
using namespace std;

/**
 * Notes:
 * [1] This program heavily uses matrices operations to get things done, which means for a better utilization,
 *      you may invert matrices in a bulky fashion
 * [2] Every function here works as if there're many matrices(e.g. depth)
 *
 */

class MatricesInverter
{
public :
    /**
     * This function takes matrices and calculates their inverses
     *
     * @param matrices This array of shape (#matrices, dim, dim), where dim is the matrix length/width
     *
     * @return Array of shape (#matrices, dim, dim) contains the inverses
     */
    xarray<double>
    invertMatrices(const xarray<double> &);

private:
    /**
     * Decompose the given matrices using Crout decomposition
     *
     * @param matrices Array of matrices to be decomposed
     * @param depth Count of the matrices
     * @param dim Size of the matrix
     *
     * @return pair of 2 arrays {@first contains all matrices L triangles}, {@second contains all matrices U triangles}
     */
    pair<xarray<double>, xarray<double>>
    matricesLUDecomposition(const xarray<double> &matrices, size_t depth, size_t dim);

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
    getLColSubtractedValue(const xarray<double> &L, const xarray<double> &U, int LColIdx, int dim);

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
    getURowSubtractedValue(const xarray<double> &L, const xarray<double> &U, int URowIdx, int dim);

    /**
     * Calculate the intermediate solutions of the system
     *
     * @param L
     * @param U
     * @param depth Count of the matrices
     * @param dim Size of the matrix
     *
     * @return xarray The intermediate solutions array
     */
    xarray<double>
    calculateIntermediateSolution(const xarray<double> &L, const xarray<double> &U, size_t depth, size_t dim);

    /**
     * Calculate the final solutions of the system
     *
     * @param L Lower triangle
     * @param U Upper triangle
     * @param D Intermediate solution matrix
     * @param depth Count of the matrices
     * @param dim Matrix dimension
     *
     * @return xarray The intermediate solutions array
     */
    xarray<double>
    calculateFinalSolution(const xarray<double> &L, const xarray<double> &U,
                           const xarray<double> &D, size_t depth, size_t dim);
};


#endif //XTENSORPROJECT_MATRICESINVERTER_H
