import pandas as pd

from boltzgen.task.filter.filter import Filter


def test_absolute_metrics_skips_optional_metrics_without_normalized_column():
    task = Filter.__new__(Filter)
    task.df = pd.DataFrame(
        {
            "design_iptm": [0.7],
            "design_ptm": [0.7],
            "min_design_to_target_pae": [4.0],
            "num_design": [8],
            "pass_has_x_filter": [True],
        }
    )
    task.filters = [{"feature": "has_x"}]

    task.absolute_metrics()

    assert "design_iiptm_z" not in task.df
    assert task.df.loc[0, "absolute_score"] != 0
    assert task.df.loc[0, "structure_confidence"] != 0
