"""
Writes A29_parsed.csv: A29.csv with only step 1 of main.py applied (check_values),
so "a.u." suffixes are stripped and text values become empty, nothing is removed.
"""
from check_values import fix_values
from main import BASE_DIR, INPUT_CSV, load_rows, save_rows

OUTPUT_CSV = BASE_DIR / "A29_parsed.csv"

header, rows = load_rows(INPUT_CSV)
rows, stats = fix_values(rows)
save_rows(OUTPUT_CSV, header, rows)
print(f"a.u. stripped: {stats['au_stripped']}, strings emptied: {stats['strings_emptied']}")
print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")
