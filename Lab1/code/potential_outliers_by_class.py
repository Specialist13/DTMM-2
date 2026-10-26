import pandas as pd


outliers = pd.read_csv("A29_potential_outliers.csv")
outliers["outlier_columns"] = outliers["outlier_columns"].fillna("")

by_class = outliers.groupby("class").size().rename("outlier_rows")
by_feature = (
    outliers.assign(feature=outliers["outlier_columns"].str.split(";"))
    .explode("feature")
    .query("feature != ''")
    .groupby("feature")["row_number"]
    .nunique()
    .sort_values(ascending=False)
)

print("Outlier rows by class:")
print(by_class.to_string())
print("\nOutlier rows by feature:")
print(by_feature.to_string())
