# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 09:02:13 2025

@author: noton
"""

import random
import pandas as pd


for data in ['Reaction_CO_1.5h', 'Reaction_CO_biphenyl', 'Reaction_CO_ortho', 'Reaction_CO_Cl', 
             'Reaction_CS', 'Reaction_CN', 'Reaction_2+2', 
             'Reaction_CF3', 'Reaction_CH2CF3', 'Reaction_CH2F', 'Reaction_Cy', 'Reaction_SCF3',
             'Reaction_OCH2F', 'Reaction_P', 'Reaction_Si']:
    N = []
    YL = []
    for i in range(60):
        random.seed(i)
        numbers = list(range(60))
        random.shuffle(numbers)
        df = pd.read_csv(f'../../data/data_woTL/data_60_{data}.csv')
        shuffled = df.iloc[numbers]
        PY = shuffled['Yield']
        yield_list = PY.tolist()
        N.append(numbers)
        YL.append(yield_list)

    df_result = pd.DataFrame({'selected_indices_per_step': N, 'actual_rewards_per_step': YL})
    df_result.to_csv(f'../results/random/random_{data}_result.csv', index=False)
