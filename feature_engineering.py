import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # BMI categories
    if 'bmi' in df.columns:
        df['BMI_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 24.9, 29.9, float('inf')],
            labels=['underweight', 'normal', 'overweight', 'obese']
        )

    # Smoker indicator (works for encoded or raw)
    if 'smoker' in df.columns:
        df['is_smoker'] = (df['smoker'] == 'yes').astype(int)
    elif 'smoker_yes' in df.columns:
        df['is_smoker'] = df['smoker_yes']
    else:
        df['is_smoker'] = 0

    # Diabetic indicator (optional, if column exists)
    if 'diabetic' in df.columns:
        df['is_diabetic'] = (df['diabetic'] == 'yes').astype(int)
    else:
        df['is_diabetic'] = 0

    # Region indicators (if not already encoded)
    if 'region' in df.columns:
        region_dummies = pd.get_dummies(df['region'], prefix='region')
        df = pd.concat([df, region_dummies], axis=1)

    return df
