import logging
import pandas as pd
from litepolis_database_default import DatabaseActor

def fetch_r_matrix(conversation_id: int) -> pd.DataFrame:
    """
    Build user-vote matrix for a specific conversation as a pandas DataFrame.
    
    Args:
        conversation_id: The ID of the conversation to build the matrix for
        
    Returns:
        A pandas DataFrame with users as rows, comments as columns, and vote values as cells
    """
    try:
        # Get all comments in the conversation
        comments = DatabaseActor.list_comments_by_conversation_id(conversation_id)
        comment_ids = [comment.id for comment in comments]
        
        if not comment_ids:
            logging.warning(f"No comments found for conversation {conversation_id}")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Get all votes for these comments
        vote_data = []
        for comment_id in comment_ids:
            votes = DatabaseActor.list_votes_by_comment_id(comment_id)
            for vote in votes:
                if vote.user_id is not None:
                    vote_data.append({
                        'user_id': vote.user_id,
                        'item_id': vote.comment_id,
                        'vote_value': vote.value
                    })
        
        if not vote_data:
            logging.warning(f"No votes found for conversation {conversation_id}")
            return pd.DataFrame()  # Return empty DataFrame
            
        df = pd.DataFrame(vote_data)
        
        # Pivot to matrix format
        r_matrix = df.pivot_table(index='user_id', columns='item_id', values='vote_value', fill_value=0)
        return r_matrix
    except Exception as e:
        logging.error(f"Failed to build R matrix for conversation {conversation_id}: {e}")
        raise