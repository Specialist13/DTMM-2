import json
import re
from collections import Counter
from enum import Enum
from pathlib import Path

import pandas as pd


AU_SUFFIX = re.compile(r"\s*a\.\s*u\.?\s*$", re.IGNORECASE)


class Types(Enum):
    NUMBER = "number"
    EMPTY = "empty"
    AU = "au"
    STRING = "string"


def classifyType(value):
    value = value.strip()
    if not value:
        return Types.EMPTY
    if "a.u." in value:
        return Types.AU
    try:
        float(value)
        return Types.NUMBER
    except ValueError:
        return Types.STRING


def fix_values(rows):
    if not rows:
        return [], Counter()
    data = pd.DataFrame(rows)
    features = data.iloc[:, :-1]
    types = features.map(classifyType)
    stats = Counter(types.to_numpy().ravel())
    au, text, empty = types.eq(Types.AU), types.eq(Types.STRING), types.eq(Types.EMPTY)
    stripped = features.apply(lambda column: column.str.strip().str.replace(AU_SUFFIX, "", regex=True).str.strip())
    data[data.columns[:-1]] = features.mask(au, stripped).mask(text | empty, "")
    stats.update(au_stripped=int(au.sum().sum()), strings_emptied=int(text.sum().sum()),
                 whitespace_emptied=int((empty & features.ne("")).sum().sum()))
    return data.values.tolist(), stats


def main():
    path = Path(__file__).resolve().parent.parent / "A29.csv"
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    totals = data["class"].str.strip().value_counts()
    cells = data.drop(columns="class").rename_axis("row").stack().rename("value").reset_index()
    cells.columns = ["row", "column", "value"]
    cells["class"] = cells["row"].map(data["class"].str.strip())
    cells["type"] = cells["value"].map(classifyType)
    cells["value"] = cells["value"].str.strip().mask(cells["type"].eq(Types.AU), "a.u.")
    result = {}
    for kind, group in cells[cells["type"].ne(Types.NUMBER)].groupby("type", sort=False):
        values = {}
        for value, matching in group.groupby("value", sort=False):
            counts = matching["class"].value_counts().sort_index()
            values[value or "(empty)"] = {
                "columns": matching["column"].value_counts().sort_index().to_dict(),
                "classes": {label: f"{count} ({count / totals[label] * 100:.3g}%)"
                            for label, count in counts.items()},
            }
        result[kind.value] = {"count": len(group), "values": values}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
