/* Project 3-1 Digit Recognition Parallelization
 * @authors: Brian Truong, Felix Liu
 *
 * You MUST implement the calc_min_dist() function in this file.
 *
 * You do not need to implement/use the swap(), flip_horizontal(), transpose(), or rotate_ccw_90()
 * functions, but you may find them useful. Feel free to define additional helper functions.
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h> /* This allows SSE Instrinsics to work. */
#include <omp.h>
#include "digit_rec.h"
#include "utils.h"

/* Declaring euclid */
float euclid(float *image, float *template, int t_width, float min_dist, int w, int h, int i_width);

/* Swaps the values pointed to by the pointers X and Y. */
// void swap(float *x, float *y) {
//     float tmp = *x;
//     *x = *y;
//     *y = tmp;
// }

/* Flips the elements of a square array ARR across the y-axis. */
void flip_horizontal(float *arr, int width) {
    int x, y;
    #pragma omp parallel for collapse(2)
    for (y = 0; y < width; y += 1) {
        for (x = 0; x < width/2; x += 1) {
            float tmp = arr[(y * width) + (width - x - 1)];
            arr[(y * width) + (width - x - 1)] = arr[(width * y + x)];
            arr[(width * y + x)] = tmp;
        }
    }
}

/* Transposes the square array ARR. */
void transpose(float *arr, int width) {
    int x, y;
    #pragma omp parallel for collapse(2)
    for (y = 0; y < width; y += 1) {
        for (x = 0; x < width; x += 1) {
            // int first = y * width + x;
            // int second = x * width + y;
            if (x != y && (y * width + x) < (x * width + y)) {
                float tmp = arr[y * width + x];
                arr[y * width + x] = arr[x * width + y];
                arr[x * width + y] = tmp;
            }
        }
    }
}

/* Optimized Transpose */
// void transpose(float *arr, int width) {
//     int i, j, k, l;
//     int blocksize = 100;
//     #pragma omp parallel for 
//     for(i = 0; i < (width/blocksize)*blocksize; i+=blocksize)
//         for(j = 0; j < (width/blocksize)*blocksize; j+=blocksize)
//             for(k=i;k<i+blocksize;k++)
//                 for(l=j;l<j+blocksize;l++)
//                     if (l != k) {
//                         int first = k * width + l;
//                         int second = l * width + k;
//                         if (first < second) {
//                             swap(&arr[first], &arr[second]);
//                         }
//                     }

//     for(i=0;i<(width/blocksize)*blocksize;i++) 
//         for(j=(width/blocksize)*blocksize;j<width;j++)
//             if (i != j) {
//                 int first = i * width + j;
//                 int second = j * width + i;
//                 if (first < second) {
//                     swap(&arr[first], &arr[second]);
//                 }
//             }

//     for(i=(width/blocksize)*blocksize;i<width;i++) 
//         for(j=0;j<width;j++) 
//             if (i != j) {
//                 int first = i * width + j;
//                 int second = j * width + i;
//                 if (first < second) {
//                     swap(&arr[first], &arr[second]);
//                 }
//             }
// }

// void transpose( int n, int blocksize, int *dst, int *src ) {

/* Rotates the square array ARR by 90 degrees counterclockwise. */
void rotate_ccw_90(float *arr, int width) {
    flip_horizontal(arr, width);
    transpose(arr, width);
}

/* Returns the squared Euclidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with side length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */
float calc_min_dist(float *image, int i_width, int i_height, 
    float *template, int t_width) {
    float min_dist = FLT_MAX;
    for (int i = 0; i < 8; i++) {
        if (i % 2 == 0) {
            rotate_ccw_90(template, t_width);
        } else {
            flip_horizontal(template, t_width);
        }
        #pragma omp parallel for collapse(2)
        for (int row = 0; row <= (i_height - t_width); row++) {
            for (int column = 0; column <= (i_width - t_width); column++) {
                float hold = 0.0;
                __m128 temp1, temp2, unsquared, actual, sumvec;
                sumvec = _mm_setzero_ps();
                int rowin = 0;
                while (rowin < t_width) {
                    int col = 0;
                    while (col < t_width/16 * 16) {
                        temp1 = _mm_loadu_ps(&image[rowin * i_width + row * i_width + col + column]);
                        temp2 = _mm_loadu_ps(&template[rowin * t_width + col]);
                        unsquared = _mm_sub_ps(temp1, temp2);
                        actual = _mm_mul_ps(unsquared, unsquared);
                        sumvec = _mm_add_ps(sumvec, actual);
                        temp1 = _mm_loadu_ps(&image[rowin * i_width + row * i_width + col + 4 + column]);
                        temp2 = _mm_loadu_ps(&template[rowin * t_width + col + 4]);
                        unsquared = _mm_sub_ps(temp1, temp2);
                        actual = _mm_mul_ps(unsquared, unsquared);
                        sumvec = _mm_add_ps(sumvec, actual);
                        temp1 = _mm_loadu_ps(&image[rowin * i_width + row * i_width + col + 8 + column]);
                        temp2 = _mm_loadu_ps(&template[rowin * t_width + col + 8]);
                        unsquared = _mm_sub_ps(temp1, temp2);
                        actual = _mm_mul_ps(unsquared, unsquared);
                        sumvec = _mm_add_ps(sumvec, actual);
                        temp1 = _mm_loadu_ps(&image[rowin * i_width + row * i_width + col + 12 + column]);
                        temp2 = _mm_loadu_ps(&template[rowin * t_width + col + 12]);
                        unsquared = _mm_sub_ps(temp1, temp2);
                        actual = _mm_mul_ps(unsquared, unsquared);
                        sumvec = _mm_add_ps(sumvec, actual);
                        col += 16;
                    }
                    if (t_width % 16 != 0) {
                        float extra1[16];
                        float extra2[16];
                        for (int xps = 0; xps < 16; xps++) {
                            if (col < t_width) {
                                extra1[xps] = image[rowin * i_width + row * i_width + col + column];
                                extra2[xps] = template[rowin * t_width + col];
                                col++;
                            } else {
                                extra1[xps] = 0.0;
                                extra2[xps] = 0.0;
                            }
                        }
                        temp1 = _mm_loadu_ps(extra1);
                        temp2 = _mm_loadu_ps(extra2);
                        unsquared = _mm_sub_ps(temp1, temp2);
                        actual = _mm_mul_ps(unsquared, unsquared);
                        sumvec = _mm_add_ps(sumvec, actual);
                        temp1 = _mm_loadu_ps(extra1 + 4);
                        temp2 = _mm_loadu_ps(extra2 + 4);
                        unsquared = _mm_sub_ps(temp1, temp2);
                        actual = _mm_mul_ps(unsquared, unsquared);
                        sumvec = _mm_add_ps(sumvec, actual);
                        temp1 = _mm_loadu_ps(extra1 + 8);
                        temp2 = _mm_loadu_ps(extra2 + 8);
                        unsquared = _mm_sub_ps(temp1, temp2);
                        actual = _mm_mul_ps(unsquared, unsquared);
                        sumvec = _mm_add_ps(sumvec, actual);
                        temp1 = _mm_loadu_ps(extra1 + 12);
                        temp2 = _mm_loadu_ps(extra2 + 12);
                        unsquared = _mm_sub_ps(temp1, temp2);
                        actual = _mm_mul_ps(unsquared, unsquared);
                        sumvec = _mm_add_ps(sumvec, actual);
                    }
                    rowin++;
                }
                hold += (sumvec[0] + sumvec[1] + sumvec[2] + sumvec[3]);
                if (hold < min_dist) {
                    min_dist = hold;
                }
            }
        }
        if (i % 2 != 0) {
            flip_horizontal(template, t_width);
        }
    }
    return min_dist;
}
