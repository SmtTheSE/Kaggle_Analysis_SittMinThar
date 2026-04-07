import json

def debug_notebook(path):
    with open(path, 'r') as f:
        nb = json.load(f)
    print(f"Total Cells: {len(nb['cells'])}")
    for i, cell in enumerate(nb['cells'][:10]):
        print(f"Cell {i}: Type={cell['cell_type']}, Source Lines={len(cell['source'])}")
        if cell['source']:
            print(f"  First line: {cell['source'][0][:50]}")
            if len(cell['source']) > 5:
                print(f"  Line 6: {cell['source'][5][:50]}")

if __name__ == "__main__":
    debug_notebook('fuel_price_analysis_notebook.ipynb')
