import unittest

from sklearn import datasets
import pandas as pd

from common import gpu_test

class TestCuml(unittest.TestCase):

    @gpu_test
    def test_linearn_classifier(self):
        from cuml import LinearRegression as cuLinearRegression
        import cudf as gd
        boston = datasets.load_boston()
        X, y = boston.data, boston.target
        df = pd.DataFrame(X,columns=['fea%d'%i for i in range(X.shape[1])])
        gdf = gd.from_pandas(df)
        lr = cuLinearRegression(fit_intercept=True,
                               normalize=False,
                               algorithm='svd')
        lr.fit(gdf,y)
