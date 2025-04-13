import pandas as pd
import streamlit as st
_ = pd

def prepare_grouped_data(df, user_inputs):
    """
    aggregation preserving columns
    """
    try:
        if user_inputs['group_by']:
            # Verify columns exist
            missing_cols = [col for col in user_inputs['group_by'] if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing grouping columns: {missing_cols}")
            
            # Create combined group column
            df['group'] = df[user_inputs['group_by']].astype(str).agg(' | '.join, axis=1)
            return df[['Date', 'target', 'group'] + user_inputs['group_by']]
        else:
            df['group'] = "All Items"
            return df[['Date', 'target', 'group']]
            
    except Exception as e:
        st.error(f"Error preparing grouped data: {str(e)}")
        st.stop()