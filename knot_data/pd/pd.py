from spherogram import random_link
from sage.knots.link import Link
from PIL import Image

unknots_src  = 'GoogleDeepmind_hard_unknots.csv'
dist_json    = 'GoogleDeepmind_hard_unknots_dist.json'
knots_json   = 'SeeingTheUnknot_knots.json'
unknots_json = 'SeeingTheUnknot_unknots.json'

for N in range(12, 41):
    L = random_link(
        N, # Generate random link with N crossings
        consistent_twist_regions=True
    )

    print(L)

    pd = L.PD_code() # Get PD code for new knot
    pd = [list(tup) for tup in pd] # Convert inner tuples to list
    
    # If zero in PD code, increment all elements
    if any(0 in row for row in pd):
        pd = [[e+1 for e in row] for row in pd]

    pd_str = str(pd).replace(' ', '') # Convert to str, remove whitespace

    # Plot
    L = Link(pd)
    p = L.plot(gap=0.5, thickness=4.5, color='black')
    filename = f'35_test.png'
    p.save(f"{filename}", dpi=300)

    img = Image.open(f"{filename}")
    img_resized = img.resize((224, 224), Image.LANCZOS)
    img_resized.save(f"{filename}")