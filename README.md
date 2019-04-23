% copyright (c) 2019
% Alexandre Reeberg de Mello - alexandre.reeberg@posgrad.ufsc.br
FBTSVM


%%%%%%EXAMPLES
Example 1 - binary classification, creates an initial model and increment it in the same function.
Example 2 - binary classification, creates an initial model and increment it using different functions.
Example 3 - binary classification, creates an initial model and increment it in the same function.
Example 4 - multiclass classification, creates an initial model and increment it in the same function.
Example 5 - multiclass classification, creates an initial model and increment it using different functions.


%%%%%%iFBTSVM parameters usage
Parameter.CC = int value (eg. 8) %C1
Parameter.CC2= int value (eg. 8)  %C3
Parameter.CR = int value (eg. 2) %C2
Parameter.CR2= int value (eg. 2)  %C4
Parameter.eps= int value (eg. 0.0001)%epsilon to avoid inverse matrix calculation error
Parameter.maxeva= int value (eg. 500) %maximum of function evaluations to each train/update the model
Parameter.u= int value (eg.0.01) %fuzzy parameter
Parameter.epsilon= int value (eg. 1e-10) %fuzzy epsilon
Parameter.repetitions= int value (eg. 3) %number of occurrences to forget
Parameter.phi= int value (eg. 0.001) %forgeting threshold
Parameter.kernel_name= char (eg. 'rbf') %kernel_name
Parameter.kernel_param= int value (eg. 0.2) %kernel parameter
Parameter.feat_dimensionality= int value (eg. 5000) %feature's dimensionality
Parameter.Napp= int value (eg. 6000) %Number of samples for the approximation
Parameter.options= char or empty (eg. 'signals' or []) %option 
Parameter.sliv= binary value (true or false) %to enable sliced variables 
ini_size= int value (eg. 0.1) %Initial training size
batch_size= float value (eg. 200 or 0.5); %int for absolute size or float to data percentage

%%%%%%SAVE AND LOAD
To save and load a model, use the default Matlab functions 'save' and 'load' respectively.
https://www.mathworks.com/help/matlab/ref/save.html
https://www.mathworks.com/help/matlab/ref/load.html


%%%%%%KERNEL APPROXIMATION
The kernel approximation method is refers to Fuxin Li, Catalin Ionescu and Cristian Sminchisescu. Random Fourier Approximations for Skewed Multiplicative Histogram Kernels. In Springer LNCS 6376. Proceedings of 32nd DAGM Symposium. More information can be found in the Randfeat_releasever folder.

kernel_name : 'linear', 'rbf' (Gaussian), 'laplace' (Laplacian),'chi2' (Chi-square), 'chi2_skewed' (Skewed chi-square), 'intersection', (Histogram intersection), 'intersection_skewed' (Skewed intersection).
kernel_param : parameter for the kernel
dim : dimensionality of the features
Napp : number of samples for the approximation
options: options. Includes: 'sampling' or signals' 
sampling for [Rahimi and Recht 2007] type for Monte Carlo sampling. 
signals for [Vedaldi and Zisserman 2010] type of fixed interval sampling. 


%%%%%%DATASETS
Example 1 and 2
Characteristics: Binary classification
Training set: 6000 instances with 5000 attributes
Test set: 1000 instances with 5000 attributes
More details @https://archive.ics.uci.edu/ml/datasets/Gisette
References: Johnson, B., Xie, Z., 2013. Classifying a high resolution image of an urban area using super-object information. ISPRS Journal of Photogrammetry and Remote Sensing, 83, 40-49. 
Johnson, B., 2013. High resolution urban land cover classification using a competitive multi-scale object-based approach. Remote Sensing Letters, 4 (2), 131-140.

Example 3
Characteristics: Binary classification
Training set: 1269 instances with 163 attributes
Test set: 744 instances with 163
More details @https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
References:Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L., A Dataset for Breast Cancer Histopathological Image Classification, IEEE Transactions on Biomedical Engineering (TBME), 63(7):1455-1462, 2016.
Mahotas, library to exctract characteristics with PFTAS.
Coelho, L.P. 2013. Mahotas: Open source software for scriptable computer vision. Journal of Open Research Software 1(1):e3, DOI: http://dx.doi.org/10.5334/jors.ac
Parameter Free Threshold Adjacency Statistics
Coelho L.P. et al. (2010) Structured Literature Image Finder: Extracting Information from Text and Images in Biomedical Literature. In: Blaschke C., Shatkay H. (eds) Linking Literature, Information, and Knowledge for Biology. Lecture Notes in Computer Science, vol 6004. Springer, Berlin, Heidelberg
Open Tracking

Example 4 and 5
Characteristics: Multiclass classification
Training set: 4000 instances with 2 attributes
Test set: 1000 instances with 2 attributes
More details @https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/






