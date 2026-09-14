import gymnasium as gym

def harvest_wins(target_wins=20):
    env = gym.make("Solitaire-v0") # Assuming your GameLogger wrapper is attached
    wins_collected = 0
    episodes_played = 0
    
    while wins_collected < target_wins:
        obs, info = env.reset()
        done = False
        
        while not done:
            # Extract valid action IDs from your observation dictionary / action mask
            valid_actions = np.where(obs['action_mask'] == 1)[0]
            
            action = greedy_klondike_policy(env, obs, valid_actions)
            
            if action is None:
                break # Deadlocked, end the episode early
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Assuming your environment passes a 'is_success' flag in the info dict
            if done and info.get('is_success', False):
                wins_collected += 1
                print(f"Win {wins_collected}/{target_wins} collected! (Episode {episodes_played})")
                
        episodes_played += 1
        if episodes_played % 1000 == 0:
            print(f"Played {episodes_played} games. Current wins: {wins_collected}")

    print("Harvesting complete.")