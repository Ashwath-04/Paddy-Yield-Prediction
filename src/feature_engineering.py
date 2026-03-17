def create_features(df):
    df['total_fertilizer'] = (
        df['DAP_20days'] +
        df['Urea_40Days'] +
        df['Potassh_50Days']
    )

    return df