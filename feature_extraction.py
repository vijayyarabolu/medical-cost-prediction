import pandas as pd


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    def categorize_bmi(bmi: float) -> str:
        if bmi < 18.5:
            return "underweight"
        if bmi < 25:
            return "normal"
        if bmi < 30:
            return "overweight"
        return "obese"

    df["bmi_category"] = df["bmi"].apply(categorize_bmi)
    return df


def add_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "smoker" in df.columns:
        df["is_smoker"] = (df["smoker"] == "yes").astype(int)
    else:
        df["is_smoker"] = 0

    if "bmi" in df.columns:
        df["is_obese"] = (df["bmi"] >= 30).astype(int)
    else:
        df["is_obese"] = 0

    if "children" in df.columns:
        df["has_children"] = (df["children"] > 0).astype(int)
    else:
        df["has_children"] = 0

    if "age" in df.columns:
        df["senior"] = (df["age"] >= 60).astype(int)
    else:
        df["senior"] = 0

    return df


def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    def age_group(age: int) -> str:
        if age < 18:
            return "child"
        if age < 30:
            return "young_adult"
        if age < 45:
            return "adult"
        if age < 60:
            return "middle_aged"
        return "senior"

    df["age_group"] = df["age"].apply(age_group)
    return df


def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = add_bmi_category(df)
    df = add_risk_indicators(df)
    df = add_demographic_features(df)
    return df


if __name__ == "__main__":
    data_path = "cleaned_insurance_data.csv"
    df = pd.read_csv(data_path)
    df = build_feature_pipeline(df)
    df.to_csv("engineered_insurance_features.csv", index=False)
