import numpy as np

def greedy_klondike_policy(env, obs, valid_actions):
    """
    valid_actions: A list of legal action integers (0 to 589) based on the env's action mask.
    We assume you have helper functions (e.g., `is_undo_action(a)`) to decode your action space.
    """
    
    # RULE 1: Never step backward. 
    # Strip out actions 562-589 (Foundation -> Tableau) to prevent infinite loops.
    forward_actions = [a for a in valid_actions if not is_undo_action(a)]
    
    if not forward_actions:
        # The only legal moves are undos. The game is deadlocked.
        return None 

    action_scores = {}
    
    for a in forward_actions:
        score = 0
        
        # PRIORITY 1: Reveal Hidden Cards (+100)
        # Moving a tableau card (or stack) to another column/foundation, resulting 
        # in a face-down card being flipped. This is the most critical move in the game.
        if reveals_hidden_card(env, a):
            score += 100
            
        # PRIORITY 2: "Safe" Foundation Moves (+75)
        # Moving a card to the foundation. (A smarter heuristic checks if the cards of the 
        # opposite color and one rank lower are already out of the deck, making it "safe").
        elif is_foundation_move(a):
            score += 75
            
        # PRIORITY 3: King to an Empty Column (+50)
        # Unjams the board and opens space to consolidate other columns.
        elif is_king_to_empty_column(env, a):
            score += 50
            
        # PRIORITY 4: Consolidate Tableau (+25)
        # Standard Tableau -> Tableau moves that don't reveal a card, but help build runs.
        elif is_tableau_to_tableau(a):
            score += 25
            
        # PRIORITY 5: Draw from Stock (+1)
        # The lowest priority. Only draw if nothing useful can be done on the board.
        elif is_draw_action(a):
            score += 1

        action_scores[a] = score

    # RULE 2: Break ties randomly.
    # If there are multiple actions with the highest score, pick one at random.
    # This ensures the solver explores different paths across multiple games.
    max_score = max(action_scores.values())
    best_actions = [a for a, s in action_scores.items() if s == max_score]
    
    return np.random.choice(best_actions)