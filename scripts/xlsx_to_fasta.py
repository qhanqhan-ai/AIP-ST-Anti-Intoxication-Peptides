#!/usr/bin/env python3
"""Convert the first worksheet in an XLSX file to FASTA without third-party packages."""

import re
import sys
import zipfile
import xml.etree.ElementTree as ET


def load_shared_strings(zf):
    try:
        with zf.open("xl/sharedStrings.xml") as handle:
            root = ET.fromstring(handle.read())
        ns = {"n": root.tag.split("}")[0].strip("{")}
        return ["".join(t.text or "" for t in si.findall(".//n:t", ns)) for si in root.findall("n:si", ns)]
    except KeyError:
        return []


def col_to_idx(col):
    value = 0
    for char in col:
        value = value * 26 + (ord(char.upper()) - 64)
    return value - 1


def get_rows_first_sheet(zf, shared_strings):
    sheet_path = "xl/worksheets/sheet1.xml"
    if sheet_path not in zf.namelist():
        candidates = [name for name in zf.namelist() if name.startswith("xl/worksheets/") and name.endswith(".xml")]
        sheet_path = sorted(candidates)[0]

    with zf.open(sheet_path) as handle:
        root = ET.fromstring(handle.read())

    ns = {"n": root.tag.split("}")[0].strip("{")}
    rows = []
    for row in root.findall(".//n:sheetData/n:row", ns):
        values = {}
        for cell in row.findall("n:c", ns):
            address = cell.get("r") or "A1"
            col = "".join(char for char in address if char.isalpha())
            cell_type = cell.get("t")
            value = ""
            if cell_type == "s":
                node = cell.find("n:v", ns)
                if node is not None and node.text:
                    idx = int(node.text)
                    value = shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
            elif cell_type == "inlineStr":
                node = cell.find(".//n:t", ns)
                value = (node.text or "") if node is not None else ""
            else:
                node = cell.find("n:v", ns)
                value = (node.text or "") if node is not None else ""
            values[col] = value
        rows.append(values)
    return rows


def get_cell(row, idx):
    for key, value in row.items():
        if col_to_idx(key) == idx:
            return value
    return ""


def main():
    if len(sys.argv) < 3:
        print("Usage: xlsx_to_fasta.py input.xlsx output.fasta")
        sys.exit(1)

    xlsx, fasta = sys.argv[1], sys.argv[2]
    with zipfile.ZipFile(xlsx) as zf:
        shared_strings = load_shared_strings(zf)
        rows = get_rows_first_sheet(zf, shared_strings)

    header = next((row for row in rows if row), None)
    if header is None:
        sys.exit("The worksheet is empty.")

    max_col = max(col_to_idx(col) for col in header.keys())
    names = []
    for idx in range(max_col + 1):
        name = next((header[col] for col in header if col_to_idx(col) == idx), "")
        names.append(name.strip().lower())

    query_idx = names.index("query") if "query" in names else None
    target_idx = names.index("target") if "target" in names else None
    if query_idx is None:
        sys.exit("Column 'query' was not found.")

    valid = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZOU*\-]+$", re.IGNORECASE)
    with open(fasta, "w", encoding="utf-8") as output:
        count = 0
        for row in rows[1:]:
            if not row:
                continue
            sequence = (get_cell(row, query_idx) or "").strip().replace(" ", "")
            target = (get_cell(row, target_idx) or "").strip() if target_idx is not None else ""
            if not sequence:
                continue
            if not valid.match(sequence):
                print(f"[WARN] suspicious sequence characters: {sequence}", file=sys.stderr)
            count += 1
            output.write(f">pep{count:05d}" + (f"|target={target}" if target else "") + "\n")
            output.write(sequence.upper() + "\n")

    print(f"[DONE] wrote {count} records to {fasta}")


if __name__ == "__main__":
    main()
