import json, sys

notebook_path = sys.argv[1]

# U+00E2 + U+20AC + U+0022
MOJI = chr(0xe2) + chr(0x20ac) + chr(0x22)
REPLACEMENT = chr(0x2014)  # em dash

with open(notebook_path, encoding="utf-8") as f:
    nb = json.load(f)

count = 0
for cell in nb["cells"]:
    src = cell["source"]
    if isinstance(src, list):
        new_src = []
        for line in src:
            fixed = line.replace(MOJI, REPLACEMENT)
            if fixed != line:
                count += line.count(MOJI)
            new_src.append(fixed)
        cell["source"] = new_src
    elif isinstance(src, str):
        fixed = src.replace(MOJI, REPLACEMENT)
        if fixed != src:
            count += src.count(MOJI)
        cell["source"] = fixed

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Fixed {count} mojibake sequences")
