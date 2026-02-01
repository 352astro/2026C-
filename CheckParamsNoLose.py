import pandas as pd
import numpy as np

def run_prediction():
    # ==========================================
    # 1. Load Data and Model
    # ==========================================
    print("Loading data and model parameters...")
    
    try:
        # 1. Load Model Parameters
        model_data = np.load('dwts_ranking_model.npz', allow_pickle=True)
        params = model_data['params']
        
        # 2. Load Feature Data
        df_features = pd.read_csv('2026_MCM_Problem_Fame_withBallroom_Data.csv')
        
        # 3. Load Score/Name Data
        df_scores = pd.read_csv('2026_MCM_Problem_C_Data.csv')
        
        # Verify row counts match
        if len(df_features) != len(df_scores):
            print(f"Warning: Row counts do not match! Features: {len(df_features)}, Scores: {len(df_scores)}")
            return

        # Merge strictly by index
        df_scores['index'] = df_features['index'] 
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        return

    # ==========================================
    # 2. Feature Engineering
    # ==========================================
    print("Processing features...")
    
    pop_raw = df_features['celebrity_fame_1'].values
    ballroom_raw = df_features['ballroom_fame_1'].values
    age_raw = df_features['celebrity_age_during_season'].values
    
    # Industry one-hot encoding columns
    ind_cols = [c for c in df_features.columns if 'industry_' in c]
    Ind_vals = df_features[ind_cols].values
    
    # Standardization Function
    def scale(x): 
        return (x - x.mean()) / (x.std() + 1e-6)
    
    pop_scaled = scale(pop_raw)
    ballroom_scaled = scale(ballroom_raw)
    age_scaled = scale(age_raw)
    
    X = np.column_stack([pop_scaled, ballroom_scaled, age_scaled, Ind_vals])
    
    # ==========================================
    # 3. Parse Parameters & Calculate Latents
    # ==========================================
    n_f = X.shape[1]
    idx = 0
    
    w_mu = params[idx : idx + n_f]
    idx += n_f
    b_mu = params[idx]
    idx += 1
    
    w_sig = params[idx : idx + n_f]
    idx += n_f
    b_sig = params[idx]
    idx += 1
    
    # Calculate Latent Variables for every dancer
    mu_values = np.dot(X, w_mu) + b_mu
    
    y2 = np.dot(X, w_sig) + b_sig
    sigma_values = np.log1p(np.exp(np.clip(y2, -20, 20))) + 1e-4
    
    # Attach parameters to dataframe
    df_scores['player_mu'] = mu_values
    df_scores['player_sigma'] = sigma_values
    
    # ==========================================
    # 4. Calculate Season Average (Imputation Value)
    # ==========================================
    # Gather all potential judge columns to calculate the season average for each dancer
    all_judge_cols = []
    for w in range(1, 12):
        for j in range(1, 5):
            all_judge_cols.append(f'week{w}_judge{j}_score')
            
    valid_all_cols = [c for c in all_judge_cols if c in df_scores.columns]
    
    # Convert all scores to numeric, coerce errors, treat 0 as NaN for averaging purposes
    # (assuming 0 means missing/eliminated in the source data context)
    temp_scores_matrix = df_scores[valid_all_cols].apply(pd.to_numeric, errors='coerce').replace(0, np.nan)
    
    # Calculate the season average for every dancer (ignoring NaNs)
    df_scores['season_global_avg'] = temp_scores_matrix.mean(axis=1)
    
    # Fill any remaining NaNs with overall dataset mean (failsafe)
    overall_mean = df_scores['season_global_avg'].mean()
    df_scores['season_global_avg'].fillna(overall_mean, inplace=True)

    # ==========================================
    # 5. Weekly Prediction Loop (No Elimination)
    # ==========================================
    print("Generating weekly predictions (including eliminated weeks)...")
    np.random.seed(2026) 
    
    output_rows = []
    seasons = sorted(df_scores['season'].unique())
    
    for season in seasons:
        season_df = df_scores[df_scores['season'] == season].copy()
        
        # Iterate Weeks 1 to 11
        for week_num in range(1, 12):
            
            # --- Logic to determine Judge Score for this week ---
            judge_cols = [f'week{week_num}_judge{j}_score' for j in range(1, 5)]
            existing_cols = [c for c in judge_cols if c in season_df.columns]
            
            if existing_cols:
                # Calculate actual average for this week
                week_scores = season_df[existing_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(0)
            else:
                # If columns don't exist (e.g. week 11 in early seasons), scores are 0
                week_scores = pd.Series([0]*len(season_df), index=season_df.index)
            
            # Apply Logic: If actual > 0, use actual. Else use season_global_avg.
            season_df['simulated_judge_score'] = np.where(
                week_scores > 0, 
                week_scores, 
                season_df['season_global_avg']
            )

            # --- Simulation Step ---
            # Iterate through ALL dancers in the season, regardless of status
            for _, row in season_df.iterrows():
                mu = row['player_mu']
                sigma = row['player_sigma']
                
                # Sample performance
                sampled_val = np.random.normal(loc=mu, scale=sigma)
                
                # Calculate Vote: Exp(S) * 1000
                vote_val = np.exp(sampled_val) * 1000
                
                output_rows.append({
                    'season': row['season'],
                    'week': f'Week_{week_num}',
                    'index': row['index'],
                    'celebrity_name': row['celebrity_name'],
                    'v_votes_exp': vote_val,
                    'judge_avg_score': row['simulated_judge_score'] # This contains imputed value if eliminated
                })

    # ==========================================
    # 6. Export
    # ==========================================
    df_output = pd.DataFrame(output_rows)
    
    # Define Column Order
    final_cols = ['season', 'week', 'index', 'celebrity_name', 'v_votes_exp', 'judge_avg_score']
    df_output = df_output[final_cols]
    
    outfile = 'predicted_fan_vote_no_loss.csv'
    df_output.to_csv(outfile, index=False)
    
    print("-" * 50)
    print(f"Prediction Complete.")
    print(f"Output saved to: {outfile}")
    print(f"Total rows generated: {len(df_output)}")
    print(f"Columns: {list(df_output.columns)}")
    print("-" * 50)
    print(df_output.head())

if __name__ == "__main__":
    run_prediction()