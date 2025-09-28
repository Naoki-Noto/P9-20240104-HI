# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 14:28:05 2025

@author: noyor
"""

import pandas as pd
import ast

model_list = [
    "BOsearch_UCB",
    "RFsearch",
    "TrABsearch_top3_e0_ne5",
    ]

All_data_names = [
    'Reaction_OCH2F',
    'Reaction_P',
    'Reaction_Si'
    ]

for model in model_list:
    for data in All_data_names:
        if model == "BOsearch_UCB":
            file_name = f'BOsearch_{data}_result'
            df = pd.read_csv(f"../results/BOsearch_UCB/{file_name}.csv")
        elif model == "TrABsearch_top3_e0_ne5":
            file_name = f'TrABsearch_{data}_top3_e0_ne5_result'
            df = pd.read_csv(f"../results/TrABsearch_top3_e0_ne5/{file_name}.csv")
        else:
            file_name = f'{model}_{data}_result'
            df = pd.read_csv(f"../results/{model}/{file_name}.csv")
            
        rewards_list = [
            ast.literal_eval(s) if isinstance(s, str) else s
            for s in df['actual_rewards_per_step']
        ]
        
        out_df = pd.DataFrame({
            'Run':   range(len(rewards_list)),
            'Round': [r.index(max(r)) + 1 for r in rewards_list]
        })
        
        out_df.to_csv(f'./results/summary_OPSsearch/{model}/{file_name}_screening.csv', index=False)
      
        
      
model = "TrABsearch_all_e0_ne5"
data = 'Reaction_Si'

file_name = f'TrABsearch_{data}_top12_e0_ne5_result'
df = pd.read_csv(f"../results/TrABsearch_all_e0_ne5/{file_name}.csv")

rewards_list = [
    ast.literal_eval(s) if isinstance(s, str) else s
    for s in df['actual_rewards_per_step']
]

out_df = pd.DataFrame({
    'Run':   range(len(rewards_list)),
    'Round': [r.index(max(r)) + 1 for r in rewards_list]
})

out_df.to_csv(f'./results/summary_OPSsearch/{model}/{file_name}_screening.csv', index=False)



model_list = [
    "BOsearch_UCB",
    "RFsearch",
    "TrABsearch_all_e0_ne5"
    ]
data = 'Reaction_CO_biphenyl'

for model in model_list:
    if model == "BOsearch_UCB":
        file_name = f'BOsearch_{data}_result'
        df = pd.read_csv(f"../results/BOsearch_UCB/{file_name}.csv")
    elif model == "TrABsearch_all_e0_ne5":
        file_name = f'TrABsearch_{data}_top14_e0_ne5_result'
        df = pd.read_csv(f"../results_others/TrABsearch_all_e0_ne5/{file_name}.csv")
    else:
        file_name = f'{model}_{data}_result'
        df = pd.read_csv(f"../results/{model}/{file_name}.csv")
        
    rewards_list = [
        ast.literal_eval(s) if isinstance(s, str) else s
        for s in df['actual_rewards_per_step']
    ]
    
    out_df = pd.DataFrame({
        'Run':   range(len(rewards_list)),
        'Round': [r.index(max(r)) + 1 for r in rewards_list]
    })
    
    out_df.to_csv(f'./results/summary_OPSsearch/{model}/{file_name}_screening.csv', index=False)

