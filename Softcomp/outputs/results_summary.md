| dataset         | classifier   | method            |   accuracy |   f1_macro |   n_features |   noise_selected |
|:----------------|:-------------|:------------------|-----------:|-----------:|-------------:|-----------------:|
| Digits+40 noise | KNN          | All features      |      0.85  |      0.848 |          104 |               40 |
| Digits+40 noise | KNN          | MI top-k          |      0.944 |      0.942 |           41 |                0 |
| Digits+40 noise | KNN          | RFE               |      0.944 |      0.943 |           41 |                2 |
| Digits+40 noise | KNN          | RF importance     |      0.956 |      0.955 |           41 |                0 |
| Digits+40 noise | KNN          | Genetic Algorithm |      0.85  |      0.843 |           41 |               14 |
| Digits+40 noise | LogReg       | All features      |      0.917 |      0.916 |          104 |               40 |
| Digits+40 noise | LogReg       | MI top-k          |      0.944 |      0.943 |           37 |                0 |
| Digits+40 noise | LogReg       | RFE               |      0.933 |      0.931 |           37 |                1 |
| Digits+40 noise | LogReg       | RF importance     |      0.956 |      0.956 |           37 |                0 |
| Digits+40 noise | LogReg       | Genetic Algorithm |      0.878 |      0.878 |           37 |               11 |
| Digits+40 noise | SVM          | All features      |      0.928 |      0.927 |          104 |               40 |
| Digits+40 noise | SVM          | MI top-k          |      0.967 |      0.966 |           40 |                0 |
| Digits+40 noise | SVM          | RFE               |      0.95  |      0.949 |           40 |                2 |
| Digits+40 noise | SVM          | RF importance     |      0.967 |      0.966 |           40 |                0 |
| Digits+40 noise | SVM          | Genetic Algorithm |      0.911 |      0.91  |           40 |               11 |
| Synthetic-60    | KNN          | All features      |      0.813 |      0.813 |           60 |               46 |
| Synthetic-60    | KNN          | MI top-k          |      0.9   |      0.899 |           11 |                2 |
| Synthetic-60    | KNN          | RFE               |      0.893 |      0.893 |           11 |                4 |
| Synthetic-60    | KNN          | RF importance     |      0.933 |      0.933 |           11 |                0 |
| Synthetic-60    | KNN          | Genetic Algorithm |      0.893 |      0.893 |           11 |                5 |
| Synthetic-60    | LogReg       | All features      |      0.853 |      0.853 |           60 |               46 |
| Synthetic-60    | LogReg       | MI top-k          |      0.873 |      0.873 |           21 |               10 |
| Synthetic-60    | LogReg       | RFE               |      0.873 |      0.873 |           21 |               13 |
| Synthetic-60    | LogReg       | RF importance     |      0.86  |      0.86  |           21 |                7 |
| Synthetic-60    | LogReg       | Genetic Algorithm |      0.853 |      0.853 |           21 |               15 |
| Synthetic-60    | SVM          | All features      |      0.9   |      0.899 |           60 |               46 |
| Synthetic-60    | SVM          | MI top-k          |      0.907 |      0.906 |           19 |                9 |
| Synthetic-60    | SVM          | RFE               |      0.88  |      0.879 |           19 |               12 |
| Synthetic-60    | SVM          | RF importance     |      0.92  |      0.919 |           19 |                5 |
| Synthetic-60    | SVM          | Genetic Algorithm |      0.887 |      0.886 |           19 |               11 |
| Wine+40 noise   | KNN          | All features      |      0.926 |      0.931 |           53 |               40 |
| Wine+40 noise   | KNN          | MI top-k          |      0.963 |      0.966 |           18 |                6 |
| Wine+40 noise   | KNN          | RFE               |      0.926 |      0.929 |           18 |                6 |
| Wine+40 noise   | KNN          | RF importance     |      0.981 |      0.983 |           18 |                5 |
| Wine+40 noise   | KNN          | Genetic Algorithm |      0.889 |      0.888 |           18 |               12 |
| Wine+40 noise   | LogReg       | All features      |      0.981 |      0.983 |           53 |               40 |
| Wine+40 noise   | LogReg       | MI top-k          |      0.981 |      0.983 |           14 |                2 |
| Wine+40 noise   | LogReg       | RFE               |      1     |      1     |           14 |                3 |
| Wine+40 noise   | LogReg       | RF importance     |      1     |      1     |           14 |                1 |
| Wine+40 noise   | LogReg       | Genetic Algorithm |      1     |      1     |           14 |                6 |
| Wine+40 noise   | SVM          | All features      |      0.944 |      0.947 |           53 |               40 |
| Wine+40 noise   | SVM          | MI top-k          |      1     |      1     |           10 |                0 |
| Wine+40 noise   | SVM          | RFE               |      1     |      1     |           10 |                1 |
| Wine+40 noise   | SVM          | RF importance     |      1     |      1     |           10 |                0 |
| Wine+40 noise   | SVM          | Genetic Algorithm |      0.907 |      0.909 |           10 |                5 |