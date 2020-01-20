RandFeat

copyright (c) 2010 
Fuxin Li - fuxin.li@ins.uni-bonn.de
Catalin Ionescu - catalin.ionescu@ins.uni-bonn.de
Cristian Sminchisescu - cristian.sminchisescu@ins.uni-bonn.de

Please check DEMO.m and DEMO_classification.m for how to use the code. DEMO.m tests the approximation against the original kernel, while DEMO_classification tries to use the random Fourier features to train linear classifiers.

Please refer to the paper:

Fuxin Li, Catalin Ionescu and Cristian Sminchisescu. Random Fourier Approximations for Skewed Multiplicative Histogram Kernels. In Springer LNCS 6376. Proceedings of 32nd DAGM Symposium.

The code should work in recent MATLAB versions on all the platforms. If it doesn't work in your platform, please send an email to us and we'll try to fix it.

We included a LIBLINEAR binary on 64-bit linux platform to test the classification performance of the code, as well as a few utility mex files to speed-up the computation of the original kernel matrix. However, you can also use your own classification algorithm to test it.

LIBLINEAR is maintained by the Machine Learning Group at National Taiwan University. Please find information in:
http://www.csie.ntu.edu.tw/~cjlin/liblinear/

Our utility mex files adapted some of Christoph Lampert's code for fast-computation of the chi-square kernel. Please
refer to:
http://sites.google.com/a/christoph-lampert.com/work/software
