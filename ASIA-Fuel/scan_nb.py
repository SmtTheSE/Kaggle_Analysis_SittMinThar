import json

with open('fuel_price_analysis_notebook.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for j, line in enumerate(cell['source']):
            if 'html' in line.lower() or '<h2' in line.lower() or '📊' in line or '\ud83d\udcca' in line:
                print(f"BINGO! Found HTML/emoji in Code Cell {i}, Line {j+1}:")
                print(f"  {line.strip()}")
