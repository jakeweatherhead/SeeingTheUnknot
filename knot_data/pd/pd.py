from spherogram import random_link
from sage.knots.link import Link
from PIL import Image
import json
import random

# See run_pd.sh for the script used to run this code

unknots_src  = 'GoogleDeepmind_hard_unknots.csv'
dist_json    = 'GoogleDeepmind_hard_unknots_dist.json'
knots_json   = 'SeeingTheUnknot_knots.json'
unknots_json = 'SeeingTheUnknot_unknots.json'

LOW_NC = 12 # NC: number of crossings
UPPER_NC = 40
CROSSING_GAP = 0.3
STRAND_THICKNESS = 1.5
CONTRIBUTION_LIMIT = 40_000 # Maximum contribution of any DCC to diagram dataset

with open(dist_json, "r") as json_f:
    hard_unknots = json.load(json_f)

hard_unknots = {
    int(num_crossings): int(num_pd_codes)
    for num_crossings, num_pd_codes in hard_unknots.items()
    if LOW_NC <= num_crossings \
        and num_crossings <= UPPER_NC
}

for N in range(LOW_NC, UPPER_NC+1):
    contribution = 0
    for sample_id in range(hard_unknots[N]+1):
        alternating = random.choice([True, False])

        L = random_link(
            crossings=N,
            alternating=alternating, 
            consistent_twist_regions=True,
            max_tries=1_000
        )

        pd = L.PD_code()
        pd = [list(tup) for tup in pd]
        
        if any(0 in row for row in pd):
            pd = [[e+1 for e in row] for row in pd]

        pd_str = str(pd).replace(' ', '')

        L = Link(pd)
        p = L.plot(
            gap=CROSSING_GAP, 
            thickness=STRAND_THICKNESS, 
            color='black'
        )
        temp_file = f'temp.png'
        p.save(f"{temp_file}", dpi=300)

        img = Image.open(f"{temp_file}")
        img_resized = img.resize((224, 224), Image.LANCZOS)
        filename = f"{N}{'a' if alternating else 'n'}{sample_id}.png"
        img_resized.save(f"../diagram/{N}/{filename}")
        contribution += 1

        if contribution == CONTRIBUTION_LIMIT:
            break