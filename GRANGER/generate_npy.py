import pandas as pd
import numpy as np
from models.utils import get_origin_expression_data

def time_rank(expression_path,PseudoTime_path):
    expression_df = pd.read_csv(expression_path, index_col=0)
    pseudotime_df = pd.read_csv(PseudoTime_path, index_col=0)
    # pseudotime_df['valid_time'] = pseudotime_df['PseudoTime'] 
    pseudotime_df['valid_time'] = pseudotime_df['PseudoTime1'].fillna(pseudotime_df['PseudoTime2'])
    # pseudotime_df['valid_time'] = pseudotime_df['PseudoTime1'].fillna(pseudotime_df['PseudoTime2']).fillna(pseudotime_df['PseudoTime3']).fillna(pseudotime_df['PseudoTime4'])
    sorted_pseudotime_df = pseudotime_df.sort_values(by='valid_time')
    sorted_cellids = sorted_pseudotime_df.index.tolist()
    sorted_expression_df = expression_df[sorted_cellids]
    sorted_expression_df.to_csv('example_data/mCAD-2000-1/time.csv')
    return sorted_expression_df


time_rank('example_data/mCAD-2000-1/ExpressionData.csv',
          'example_data/mCAD-2000-1/PseudoTime.csv')

def to_npy(expression_path):
    a,b=get_origin_expression_data(expression_path)
    arr = np.vstack(list(a.values()))
    np.save('example_data/mCAD-2000-1/time_output.npy', arr)
to_npy('example_data/mCAD-2000-1/time.csv')