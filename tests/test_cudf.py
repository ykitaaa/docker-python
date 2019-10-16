import unittest
import pandas as pd

from common import gpu_test

class TestCudf(unittest.TestCase):
 
    @gpu_test   
    def test_read_csv(self):
        import cudf as gd
        cols = ['label'] + ['pixel%d'%i for i in range(784)]
        dtypes = ['int32' for i in cols]
        path = "/input/tests/data/train.csv"
        data_gd = gd.read_csv(path,names=cols,dtype=dtypes,skiprows=1)
        data_pd = pd.read_csv(path)
        self.assertEqual(100, len(data_gd.index))
        for col in data_gd.columns:
            col_equal = data_pd[col].values == data_gd[col].to_pandas().values
            self.assertEqual(True, min(col_equal))

    @gpu_test
    def test_groupby(self):
        import cudf as gd
        cols = ['label'] + ['pixel%d'%i for i in range(784)]
        dtypes = ['int32' for i in cols]
        path = "/input/tests/data/train.csv"
        data_gd = gd.read_csv(path,names=cols,dtype=dtypes,skiprows=1)

        dg = data_gd.groupby('pixel0').agg({'label':['mean','max','min']})
        dg.columns = ['%s_label'%s for s in ['mean','max','min']]
        dg = dg.reset_index()
        dg = dg.to_pandas() 
        self.assertEqual(4.22, dg.mean_label.values[0])
        self.assertEqual(9, dg.max_label.values[0])
        self.assertEqual(0, dg.min_label.values[0])