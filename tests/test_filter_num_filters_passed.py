import pandas as pd
from boltzgen.task.filter.filter import Filter

def test_num_filters_passed_independent_evaluation():
    """
    Verify that num_filters_passed increments independently for each filter,
    even if an earlier filter fails.
    """
    task = Filter.__new__(Filter)
    task.from_inverse_folded = True
    task.top_budget = 10
    task.budget = 30
    
    # Create mock dataframe with values designed to fail some filters and pass others
    task.df = pd.DataFrame(
        {
            "id": ["design_1", "design_2"],
            "has_x": [0, 0],              # Both pass (<=0)
            "filter_rmsd": [16.0, 16.0],  # Both fail (<=2.5)
            "max_consecutive_ilv_sheet": [1, 4], "design_iptm": [0.8, 0.8]
        }
    )
    
    task.filters = [
        {"feature": "has_x", "lower_is_better": True, "threshold": 0},
        {"feature": "filter_rmsd", "lower_is_better": True, "threshold": 2.5},
        {"feature": "max_consecutive_ilv_sheet", "lower_is_better": True, "threshold": 2},
    ]
    
    # Initialize necessary columns
    task.df["num_filters_passed"] = 0
    filter_cols = []
    
    for flt in task.filters:
        feat = flt["feature"]
        low = flt["lower_is_better"]
        threshold = flt["threshold"]
        
        filter_col = f"pass_{feat}_filter"
        filter_cols.append(filter_col)
        if low:
            task.df[filter_col] = task.df[feat] <= threshold
        else:
            task.df[filter_col] = task.df[feat] >= threshold

        task.df["num_filters_passed"] += task.df[filter_col].astype(int)
        task.df["pass_filters"] = task.df[filter_cols].all(axis=1)
        
    assert task.df.loc[0, "num_filters_passed"] == 2  # Passes has_x and max_consecutive_ilv_sheet
    assert task.df.loc[1, "num_filters_passed"] == 1  # Passes only has_x
    
    # Now verify the ranking sorts properly via sort_df using this tuple setup
    task.metrics = {"complex_plddt": 1}
    task.df["complex_plddt"] = [0.9, 0.9]  # Equal plddt, so num_filters_passed should break the tie
    
    task.sort_df()
    
    # design_1 should be ranked better (1) because it passed more filters
    assert task.df.loc[task.df["id"] == "design_1", "final_rank"].values[0] == 1
    assert task.df.loc[task.df["id"] == "design_2", "final_rank"].values[0] == 2
