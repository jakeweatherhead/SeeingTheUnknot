from spherogram import random_link
from sage.knots.link import Link
from PIL import Image

# Input/Output Files
unknots_src  = 'GoogleDeepmind_hard_unknots.csv'
dist_json    = 'GoogleDeepmind_hard_unknots_dist.json'
knots_json   = 'SeeingTheUnknot_knots.json'
unknots_json = 'SeeingTheUnknot_unknots.json'

# Diagrammatic Crossing Count Bounds
LOWER_DCC = 12
UPPER_DCC = 40

for N in range(LOWER_DCC, UPPER_DCC+1):
    L = random_link(
        N, # Generate random link with N crossings
        consistent_twist_regions=True
    )

    print(L)
    pd = L.PD_code()
    pd = [list(tup) for tup in pd]
    
    # If zero in PD code, increment all elements for Sage compatibility
    if any(0 in row for row in pd):
        pd = [[e+1 for e in row] for row in pd]

    pd_str = str(pd).replace(' ', '')

    # Plot
    L = Link(pd)
    p = L.plot(gap=0.3, thickness=1, color='black')
    filename = f'test.png'
    p.save(f"{filename}", dpi=300)

    img = Image.open(f"{filename}")
    img_resized = img.resize((224, 224), Image.LANCZOS)
    img_resized.save(f"test+{filename}")