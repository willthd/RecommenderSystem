import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

'''
train_df : userId, itemId, rating으로 구성되어 이는 dataframe.
index가 'itemId'이면 이는 곧 item based로 계산을, 'userId'이면 user based로 계산을 하는 것을 의미.
ex) index_col='itemId', columns_col='userId', values_col='rating'
'''


def get_sparse_matrix(train_df:pd.DataFrame,
                      index_col:str,
                      columns_col:str,
                      values_col:str,
                      fillna_num:float=0) -> pd.DataFrame:
    sparse_matrix = train_df.pivot(
        index=index_col,
        columns=columns_col,
        values=values_col,
    ).fillna(fillna_num)
    
    return sparse_matrix


def get_cos_simialrity_df(sparse_matrix:pd.DataFrame) -> pd.DataFrame:
    # index기준으로 cos_similarity 계산. ex) index: userId or itemId
    cossim_values = cosine_similarity(sparse_matrix.values, sparse_matrix.values)
    cossim_df = pd.DataFrame(
        data=cossim_values,
        columns = sparse_matrix.index,
        index=sparse_matrix.index
    )
    
    return cossim_df


def predict(train_df:pd.DataFrame, item_sparse_matrix:pd.DataFrame, item_cossim_df:pd.DataFrame) -> pd.DataFrame:
    '''
    이해를 쉽게 하기 위해 item based일 때의 instance name 만듦. 만약 user based로 한다면 instance_name이 user와 item 반대가 되도록 하면 됨.
    '''
    userId_grouped = train_df.groupby('userId')
    # train_df 내에서 user가 평가한 itemId만 index로 함.
    item_prediction_result_df = pd.DataFrame(index=list(userId_grouped.indices.keys()), columns=item_sparse_matrix.index)
    item_prediction_result_df.index.name = 'userId'
    for userId, group in userId_grouped:
        # user 한명이 rating한 itemId * 전체 itemId
        choosed_item_sim = item_cossim_df.loc[group['itemId']]
        # user 한명이 rating한 itemId * 1, user가 선택한 item에 대해서만 rating
        user_rating = group['rating']
        # 전체 itemId * 1, 전체 item에 대해 개별적으로 user가 선택한 item만큼만 sum
        sim_sum = choosed_item_sim.sum(axis=0)
        # userId의 전체 rating predictions (8938 * 1)
        pred_ratings = np.matmul(choosed_item_sim.T.to_numpy(), user_rating) / (sim_sum+1)
        item_prediction_result_df.loc[userId] = pred_ratings
        
    return item_prediction_result_df.T
        
        
if __name__ == '__main__':
    # item based case
    data = pd.read_pickle('../data/mock_data.pkl')
    display(data.shape, data)
    item_sparse_matrix = get_sparse_matrix(data, 'itemId', 'userId', 'rating')
    item_cossim_df = get_cos_simialrity_df(item_sparse_matrix)
    prediction = predict(data, item_sparse_matrix, item_cossim_df)
    result = item_sparse_matrix.copy()
    result[result == 0] = prediction
    display(item_sparse_matrix, prediction, result)