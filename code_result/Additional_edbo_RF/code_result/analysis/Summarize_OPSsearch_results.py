# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:08:15 2024

@author: noyor
"""

import pandas as pd
import ast

model_list = [
    "BOsearch_EI",
    "BOsearch_PI",
    "BOsearch_UCB", 
    "LRsearch",
    "RFsearch",
    "RFsearch_e10%",
    "random",
    "TrABsearch_all_e0_ne5",
    "TrABsearch_bottom3_e0_ne5",
    "TrABsearch_top1_e0_ne5",
    "TrABsearch_top3_e0_ne5",
    "TrABsearch_top3_e0_ne3",
    "TrABsearch_top3_e0_ne10",
    "TrABsearch_top5_e0_ne5",
    "TrABsearch_top3_e0_ne5_LGBM",
    "TrABsearch_top3_e0_ne5_XGB",
    "TrABsearch_top3_e0_ne5_MF",
    "TrABsearch_top3_e10%_ne5",
    ]

All_data_names = ['Reaction_OCH2F','Reaction_P', 'Reaction_Si']

threshold_dict = {
    'Reaction_Si': 76,
    'Reaction_P': 95,
    'Reaction_OCH2F': 81,
    }


def find_positions(lst, target_indices):
    target_set = set(target_indices)
    positions = []
    found_targets = set()

    for idx, val in enumerate(lst):
        if val in target_set and val not in found_targets:
            found_targets.add(val)
            positions.append(idx + 1)
        if found_targets == target_set:
            break

    while len(positions) < len(target_indices):
        positions.append(None)

    if found_targets != target_set:
        raise ValueError("All target_indices were not selected in this run.")

    return positions


original_data_dict = {}
for target in All_data_names:
    original_data_dict[f'{target}'] = pd.read_csv(f'../../data/data_60/data_60_{target}.csv')   

for target in All_data_names:
    for model in model_list:
        if model == 'TrABsearch_all_e0_ne5':
            file_name = f'TrABsearch_{target}_top12_e0_ne5_result'                      
        elif model == 'TrABsearch_bottom3_e0_ne5':
            file_name = f'TrABsearch_{target}_bottom3_e0_ne5_result'
        elif model == 'TrABsearch_top1_e0_ne5':
            file_name = f'TrABsearch_{target}_top1_e0_ne5_result'
        elif model == 'TrABsearch_top3_e0_ne5':
            file_name = f'TrABsearch_{target}_top3_e0_ne5_result'
        elif model == 'TrABsearch_top3_e0_ne3':
            file_name = f'TrABsearch_{target}_top3_e0_ne3_result'
        elif model == 'TrABsearch_top3_e0_ne10':
            file_name = f'TrABsearch_{target}_top3_e0_ne10_result'
        elif model == 'TrABsearch_top5_e0_ne5':
            file_name = f'TrABsearch_{target}_top5_e0_ne5_result'
        elif model == 'TrABsearch_top3_e0_ne5_LGBM':
            file_name = f'TrABsearch_{target}_top3_e0_ne5_result'            
        elif model == 'TrABsearch_top3_e0_ne5_XGB':
            file_name = f'TrABsearch_{target}_top3_e0_ne5_result'
        elif model == 'TrABsearch_top3_e0_ne5_MF':
            file_name = f'TrABsearch_{target}_top3_e0_ne5_result'
        elif model == 'TrABsearch_top3_e10%_ne5':
            file_name = f'TrABsearch_{target}_top3_e0.1_ne5_result'
        elif model == 'BOsearch_EI':
            file_name = f'BOsearch_{target}_result'   
        elif model == 'BOsearch_PI':
            file_name = f'BOsearch_{target}_result'  
        elif model == 'BOsearch_UCB':
            file_name = f'BOsearch_{target}_result'  
        elif model == 'RFsearch_e10%':
            file_name = f'RFsearch_{target}_result'
        elif model == 'random':
            file_name = f'random_{target}_result'
        else:
            file_name = f'{model}_{target}_result'
        
        df = pd.read_csv(f'../results/{model}/{file_name}.csv')
        df['selected_indices_per_step'] = df['selected_indices_per_step'].apply(ast.literal_eval)
        
        target_indices = []
        data_df = original_data_dict[target]
        threshold = threshold_dict[target]
        selected_indices = data_df[data_df['Yield'] >= threshold].index.tolist()
        target_indices.extend(selected_indices)
        
        summary_df = pd.DataFrame()
        step_columns = [f'Selected_OPS_count_{i+1}' for i in range(len(target_indices))]
        summary_df[step_columns] = df['selected_indices_per_step'].apply(lambda x: pd.Series(find_positions(x, target_indices)))
        summary_df.insert(0, 'Run_index', range(60))
        summary_df.to_csv(f'./results/summary_OPSsearch/high_activity_selection/{model}/{file_name}_OPSsearch_summary.csv', index=False)
        
        mean_series = summary_df.drop(columns='Run_index').mean()
        std_series = summary_df.drop(columns='Run_index').std(ddof=0)
        columns = mean_series.index
        average_df = pd.DataFrame([mean_series.values], columns=columns)
        std_df = pd.DataFrame([std_series.values], columns=columns)
        summary_stats_df = pd.concat([average_df, std_df], ignore_index=True)
        summary_stats_df.insert(0, 'Stat', ['average', 'std'])
        summary_stats_df.to_csv(f'./results/summary_OPSsearch/stats/{model}/{file_name}_OPSsearch_stats.csv', index=False)



