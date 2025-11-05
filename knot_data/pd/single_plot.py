
from spherogram import random_link
from sage.knots.link import Link


L = random_link(
        crossings=5000,
        num_components=1,  # Prevent n-component links (n >= 2)
        alternating=True,
        consistent_twist_regions=True,
        max_tries=1_000
    )

pd = L.PD_code()

if any(0 in row for row in pd):  # for SageMath compatibility
            pd = [[e + 1 for e in row] for row in pd]

L = Link(pd)
p = L.plot(gap=0.25, thickness=1.5, color='black')
p.save(f'5000.png', dpi=300)