CPU Optimization
==

This was an extended take on matrix manipulation from an earlier project and it was to use parallelism and SIMD instructions for Intel to make image processing faster on the CPU. In this case, we only handled small to medium-sized matrices. For larger matrices, we used the GPU and the code for that can be seen in my GPU repository. The hardest part of this project was finding the bottleneck in the code and where to place OpenMP statements to attain the fastest speeds and avoid false sharing or duplicate tasks. Using a combination of OpemMP, SIMD instructions, and loop unrolling, my partner and I was able to achieve speeds around 20x faster than our original code would have performed. Below is a snippet of our minimum distance calculator with all three of these optimization implementations at work. 

```c
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
```
