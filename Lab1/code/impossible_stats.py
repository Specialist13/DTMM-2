import csv

import pandas as pd


STD_COLUMNS = range(12, 24)
EXCESS_COLUMNS = range(36, 48)


def blank_impossible_stat_values(rows):
    data = pd.DataFrame(rows, dtype="string")
    numbers = data.apply(pd.to_numeric, errors="coerce")
    masks = {
        **{column: (numbers[column] < 0, "negative standard deviation") for column in STD_COLUMNS},
        **{column: (numbers[column] < -2, "excess kurtosis below -2") for column in EXCESS_COLUMNS},
    }
    records = [
        {"row": row + 1, "column": column, "value": data.iat[row, column], "reason": reason}
        for column, (mask, reason) in masks.items()
        for row in data.index[mask.fillna(False)]
    ]
    for record in records:
        data.iat[record["row"] - 1, record["column"]] = ""
    return data.values.tolist(), records


if __name__ == "__main__":
    with open("A29.csv", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))[1:]
    print(f"impossible stat values: {len(blank_impossible_stat_values(rows)[1])}")
