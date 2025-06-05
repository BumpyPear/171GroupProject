import pandas as pd

def json_to_df(json_data: dict, feature_list: list):
    """
    1) Verifies that json_data contains *all* features in feature_list.  
    2) Verifies that none of those values are None or a non-numeric string.  
    3) Returns a single-row DataFrame with columns=feature_list in that exact order.
    Raises:
      - KeyError if any feature is missing.
      - ValueError if a feature can't be converted to float.
    """
    # 1) Check for missing keys:
    missing = [feat for feat in feature_list if feat not in json_data]
    if missing:
        raise KeyError(f"Missing required features: {missing}")

    # 2) Build a list of values (in feature_list order), coercing to float:
    row = []
    for feat in feature_list:
        val = json_data[feat]
        if val is None:
            raise ValueError(f"Feature '{feat}' is null.")
        try:
            num = float(val)
        except Exception:
            raise ValueError(f"Feature '{feat}' is not numeric: received '{val}'")
        row.append(num)

    # 3) Make a one-row DataFrame
    df = pd.DataFrame([row], columns=feature_list)
    return df
