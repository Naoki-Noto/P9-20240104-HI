# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:08:15 2024

@author: noyor
"""

import os
import random
import numpy as np
import pandas as pd
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class OPSsearch_env:
    def __init__(self, features, rewards, epsilon):
        self.features = features  
        self.rewards = rewards   
        self.remaining_indices = list(range(len(features)))       
        self.done = False
        self.count = 0         
        
    def reset(self):
        self.remaining_indices = list(range(len(self.features)))
        self.done = False
        return self.remaining_indices
    
    def select_action(self, model, epsilon):
        """
        Determining the action based on the epsilon-greedy method
        epsilon : a catalyst is randomly selected
        1-epsilon: the catalyst with the highest predicted yield is selected.
        """
        random_value = random.random()
        if random_value < epsilon:
            #print("random search", random_value)
            random_index = np.random.choice(len(self.remaining_indices))
            best_catalyst_remaining_index = random_index
            best_catalyst_actual_index = self.remaining_indices[random_index]
        else:
            predicted_rewards = model.predict(pd.DataFrame(self.get_remaining_features()))
            #print(predicted_rewards)
            predicted_rewards_1d = predicted_rewards.ravel()
            best_catalyst_remaining_index = np.argmax(predicted_rewards_1d)
            best_catalyst_actual_index = self.remaining_indices[best_catalyst_remaining_index]
    
        self.count += 1
        return best_catalyst_remaining_index, best_catalyst_actual_index
    
    def step(self, chosen_index):
        """
        Applies the chosen action and returns the next state, the actual reward, and the done flag.
        chosen_index: the actual index of the selected catalyst in the original dataset.
        """
        actual_reward = self.rewards[chosen_index]
        self.remaining_indices.remove(chosen_index)
        next_state = self.remaining_indices
        if len(self.remaining_indices) == 0:
            self.done = True

        return next_state, actual_reward, self.done
    
    def get_remaining_features(self):
        return self.features[self.remaining_indices]
    
    def get_remaining_indices(self):
        return self.remaining_indices


def RFsearch(target, n_OPS=1, epsilon=0.05, data_dir="../../data/data_woTL", result_dir="../results/RFsearch"):
    print(f'Running RFsearch for {target}...')
    file_path = os.path.join(data_dir, f"data_60_{target}.csv")
    df_data = pd.read_csv(file_path)
    features = df_data.drop(columns=['Name', 'ID', 'Yield']).values
    rewards = df_data[['Yield']].values

    output_data = []
    for i in range(n_OPS):
        initial_indices = [i]

        random.seed(42)
        np.random.seed(42)
        
        initial_features = features[initial_indices]
        initial_rewards = rewards[initial_indices]
        
        model = RandomForestRegressor(random_state=42)
        model.fit(initial_features, initial_rewards.ravel())
        
        selected_indices = initial_indices.copy()
        selected_features = initial_features.tolist()
        selected_rewards = initial_rewards.tolist()
        
        env = OPSsearch_env(features=features, rewards=rewards, epsilon=epsilon)
        
        actual_rewards_per_step = []
        predicted_rewards_per_step = []
        selected_indices_per_step = []
        for index in sorted(initial_indices, reverse=True):
            env.remaining_indices.remove(index)
        
        for step in range(59):
            if step == 0:
                random.seed(i)
                best_catalyst_actual_index = random.choice(env.get_remaining_indices())
            else:
                best_catalyst_remaining_index, best_catalyst_actual_index = env.select_action(model, epsilon)
            
            #OPS_n = best_catalyst_actual_index + 1
            best_catalyst_feature = env.get_remaining_features()[env.get_remaining_indices().index(best_catalyst_actual_index)]
            best_catalyst_feature = np.array(best_catalyst_feature).reshape(1, -1)
            predicted_reward = model.predict(best_catalyst_feature).item()
            #print(f"Step {step+3}: OPS {OPS_n} was selected (Pred Yield: {predicted_reward:.0f})")
            
            next_state, actual_reward, done = env.step(best_catalyst_actual_index)
            
            actual_rewards_per_step.append(actual_reward)
            predicted_rewards_per_step.append(predicted_reward)
            selected_indices_per_step.append(best_catalyst_actual_index)
            
            selected_indices.append(best_catalyst_actual_index)
            selected_features.append(features[best_catalyst_actual_index].tolist())
            selected_rewards.append(actual_reward)
            
            model.fit(selected_features, np.array(selected_rewards).ravel())
            
            if done:
                break
        
        output_data.append({'initial_indices': initial_indices,
                            'initial_reward': initial_rewards.item(),
                            'selected_indices_per_step': selected_indices_per_step,
                            'actual_rewards_per_step': actual_rewards_per_step,
                            'predicted_rewards_per_step': predicted_rewards_per_step})
        
    df_result = pd.DataFrame(output_data)
    df_out = pd.DataFrame()
    
    df_out['selected_indices_per_step'] = df_result['initial_indices'] + df_result['selected_indices_per_step']
    df_out['actual_rewards_per_step'] = df_result.apply(lambda row: [row['initial_reward']] + [x.item() for x in row['actual_rewards_per_step']], axis=1)
    df_out['predicted_rewards_per_step'] = df_result['predicted_rewards_per_step']
    
    out_file = os.path.join(result_dir, f"RFsearch_{target}_result.csv")
    df_out.to_csv(out_file, index=False)
    return df_out


def LRsearch(target, n_OPS=1, epsilon=0.05, data_dir="../../data/data_woTL", result_dir="../results/LRsearch"):
    print(f'Running LRsearch for {target}...')
    file_path = os.path.join(data_dir, f"data_60_{target}.csv")
    df_data = pd.read_csv(file_path)
    features = df_data.drop(columns=['Name', 'ID', 'Yield']).values
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0, ddof=1)
    nonzero_std_mask = (std != 0)
    features[:, nonzero_std_mask] = (features[:, nonzero_std_mask] - mean[nonzero_std_mask]) / std[nonzero_std_mask]
    rewards = df_data[['Yield']].values

    output_data = []
    for i in range(n_OPS):
        initial_indices = [i]

        random.seed(42)
        np.random.seed(42)
        
        initial_features = features[initial_indices]
        initial_rewards = rewards[initial_indices]
        
        model = LinearRegression()
        model.fit(initial_features, initial_rewards.ravel())
        
        selected_indices = initial_indices.copy()
        selected_features = initial_features.tolist()
        selected_rewards = initial_rewards.tolist()
        
        env = OPSsearch_env(features=features, rewards=rewards, epsilon=epsilon)
        
        actual_rewards_per_step = []
        predicted_rewards_per_step = []
        selected_indices_per_step = []
        for index in sorted(initial_indices, reverse=True):
            env.remaining_indices.remove(index)
        
        for step in range(59):
            if step == 0:
                random.seed(i)
                best_catalyst_actual_index = random.choice(env.get_remaining_indices())
            else:
                best_catalyst_remaining_index, best_catalyst_actual_index = env.select_action(model, epsilon)
            
            #OPS_n = best_catalyst_actual_index + 1
            best_catalyst_feature = env.get_remaining_features()[env.get_remaining_indices().index(best_catalyst_actual_index)]
            best_catalyst_feature = np.array(best_catalyst_feature).reshape(1, -1)
            predicted_reward = model.predict(best_catalyst_feature).item()
            #print(f"Step {step+3}: OPS {OPS_n} was selected (Pred Yield: {predicted_reward:.0f})")
            
            next_state, actual_reward, done = env.step(best_catalyst_actual_index)
            
            actual_rewards_per_step.append(actual_reward)
            predicted_rewards_per_step.append(predicted_reward)
            selected_indices_per_step.append(best_catalyst_actual_index)
            
            selected_indices.append(best_catalyst_actual_index)
            selected_features.append(features[best_catalyst_actual_index].tolist())
            selected_rewards.append(actual_reward)
            
            model.fit(selected_features, np.array(selected_rewards).ravel())
            
            if done:
                break
        
        output_data.append({'initial_indices': initial_indices,
                            'initial_reward': initial_rewards.item(),
                            'selected_indices_per_step': selected_indices_per_step,
                            'actual_rewards_per_step': actual_rewards_per_step,
                            'predicted_rewards_per_step': predicted_rewards_per_step})
        
    df_result = pd.DataFrame(output_data)
    df_out = pd.DataFrame()
    
    df_out['selected_indices_per_step'] = df_result['initial_indices'] + df_result['selected_indices_per_step']
    df_out['actual_rewards_per_step'] = df_result.apply(lambda row: [row['initial_reward']] + [x.item() for x in row['actual_rewards_per_step']], axis=1)
    df_out['predicted_rewards_per_step'] = df_result['predicted_rewards_per_step']
    
    out_file = os.path.join(result_dir, f"LRsearch_{target}_result.csv")
    df_out.to_csv(out_file, index=False)
    return df_out

