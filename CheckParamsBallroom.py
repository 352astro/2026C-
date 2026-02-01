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
        
        # 2. Load Feature Data (Contains Fame, Ballroom Fame, Age, Industry, Index)
        df_features = pd.read_csv('2026_MCM_Problem_Fame_withBallroom_Data.csv')
        
        # 3. Load Score/Name Data (Contains Season, Name, Weekly Scores)
        df_scores = pd.read_csv('2026_MCM_Problem_C_Data.csv')
        
        # Verify row counts match
        if len(df_features) != len(df_scores):
            print(f"Warning: Row counts do not match! Features: {len(df_features)}, Scores: {len(df_scores)}")
            return

        # Merge strictly by index (row alignment)
        # We assign the 'index' from features to scores for tracking
        df_scores['index'] = df_features['index'] 
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        return

    # ==========================================
    # 2. Feature Engineering (Match optimize.py)
    # ==========================================
    print("Processing features...")
    
    # Extract raw values
    # NOTE: Ensure column names match exactly what is in the new CSV
    pop_raw = df_features['celebrity_fame_1.4'].values
    ballroom_raw = df_features['ballroom_fame_1.4'].values # New Feature
    age_raw = df_features['celebrity_age_during_season'].values
    
    # Industry one-hot encoding columns
    ind_cols = [c for c in df_features.columns if 'industry_' in c]
    Ind_vals = df_features[ind_cols].values
    
    # Standardization Function (Must match training logic)
    def scale(x): 
        return (x - x.mean()) / (x.std() + 1e-6)
    
    # Create Feature Matrix X
    # Order: [Pop, Ballroom, Age, Industry...]
    pop_scaled = scale(pop_raw)
    ballroom_scaled = scale(ballroom_raw)
    age_scaled = scale(age_raw)
    
    X = np.column_stack([pop_scaled, ballroom_scaled, age_scaled, Ind_vals])
    
    # ==========================================
    # 3. Parse Parameters & Calculate Latents
    # ==========================================
    n_f = X.shape[1] # Number of features
    
    # Parameter Mapping from optimize.py structure:
    # [w_mu(n_f), b_mu(1), w_sig(n_f), b_sig(1), w_judge(1)]
    
    # Pointer index
    idx = 0
    
    # Mu (Mean Ability)
    w_mu = params[idx : idx + n_f]
    idx += n_f
    b_mu = params[idx]
    idx += 1
    
    # Sigma (Volatility)
    w_sig = params[idx : idx + n_f]
    idx += n_f
    b_sig = params[idx]
    idx += 1
    
    # Judge Weight (Not needed for vote calculation, but extracted to verify structure)
    # w_judge = params[idx] 
    
    # Calculate Latent Variables for every dancer
    # 1. Mu
    mu_values = np.dot(X, w_mu) + b_mu
    
    # 2. Sigma (Must match the softplus logic in optimize.py to prevent negative std dev)
    # Logic: sigma = log1p(exp(clip(dot + b)))
    y2 = np.dot(X, w_sig) + b_sig
    sigma_values = np.log1p(np.exp(np.clip(y2, -20, 20))) + 1e-4
    
    # Attach these to the main dataframe for easier weekly processing
    df_scores['player_mu'] = mu_values
    df_scores['player_sigma'] = sigma_values
    
    # ==========================================
    # 4. Weekly Prediction Loop
    # ==========================================
    print("Generating weekly predictions...")
    np.random.seed(2026) # Reproducibility
    
    output_rows = []
    
    # Iterate through every season present in the data
    seasons = sorted(df_scores['season'].unique())
    
    for season in seasons:
        # Filter data for current season
        season_df = df_scores[df_scores['season'] == season].copy()
        
        # Iterate Weeks 1 to 11
        for week_num in range(1, 12):
            # Identify columns for judges scores this week
            # Assuming format like 'week1_judge1_score', etc.
            judge_cols = [f'week{week_num}_judge{j}_score' for j in range(1, 5)]
            
            # Check if this week exists in columns
            existing_cols = [c for c in judge_cols if c in season_df.columns]
            if not existing_cols:
                continue
            
            # Calculate Average Judge Score for the week
            # Coerce errors to NaN, then mean ignoring NaNs
            week_scores = season_df[existing_cols].apply(pd.to_numeric, errors='coerce')
            season_df['current_avg'] = week_scores.mean(axis=1)
            
            # Identify active dancers: those who have a score > 0 this week
            active_dancers = season_df[season_df['current_avg'] > 0].copy()
            
            if active_dancers.empty:
                continue
            
            # --- Simulation Step ---
            for _, row in active_dancers.iterrows():
                # Get latent parameters
                mu = row['player_mu']
                sigma = row['player_sigma']
                
                # Sample performance/popularity for this week
                # S ~ N(mu, sigma)
                sampled_val = np.random.normal(loc=mu, scale=sigma)
                
                # Calculate Raw Vote Value (Exponential, NO Softmax)
                # V = exp(S)
                vote_val = np.exp(sampled_val) * 1000
                
                # Append result row
                output_rows.append({
                    'season': row['season'],
                    'week': f'Week_{week_num}',
                    'index': row['index'],
                    'celebrity_name': row['celebrity_name'],
                    'v_votes_exp': vote_val,
                    'judge_avg_score': row['current_avg']
                })

    # ==========================================
    # 5. Export
    # ==========================================
    df_output = pd.DataFrame(output_rows)
    
    # Define Column Order
    final_cols = ['season', 'week', 'index', 'celebrity_name', 'v_votes_exp', 'judge_avg_score']
    df_output = df_output[final_cols]
    
    outfile = 'predicted_fan_votes_v.csv'
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