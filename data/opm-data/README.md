opm-data
========

This is the OPM repository which is supposed to contain all relevant
datasets and simulation results which are required to test the OPM
simulators thoroughly.


Unless stated otherwise, decks are now provided under the open
database license available in the file odbl-10.txt, and new decks
should also be provided under the same or a compatible license.
Similarly, data contained in or pointed to by the decks is licensed
under the open database content license available in the file
dbcl-10.txt.

Note that the git history contains proprietary decks (now removed) where
usage restrictions may apply.
-----------------------------------------------------
Grid dimensions:
60 × 220 × 85 cells = 1,122,000 cells

There is 1 injection well in the centre pushes water in under high pressure, which forces the oil to move. 4 production wells at the corners simply sit at low pressure, so the oil naturally flows toward them and up to the surface.
-----------------------------------------------------
Explaining the dataset of SPE10Model2

- SPE10MODEL2_PHI.INC - porousty of each cell:

This file store the 1,122,000 porosity (φ) value of each cell, which is the space available for fluid to occupy. It's a dimensionless number between 0 and 1.

Min porosity -> 	~0.0 (dead rock / tight shale)
Max porosity ->     0.50 (very porous sand)
Mean porosity->     ~0.17 (17% — typical reservoir rock)

How values map to the grid:
value[0]       → cell (x=0, y=0, z=0)   ← top-left of layer 1
value[1]       → cell (x=1, y=0, z=0)
...
value[59]      → cell (x=59, y=0, z=0)
value[60]      → cell (x=0,  y=1, z=0)
...
value[13199]   → cell (x=59, y=219, z=0)  ← end of layer 1
value[13200]   → cell (x=0,  y=0,  z=1)   ← start of layer 2
...
value[1121999] → cell (x=59, y=219, z=84) ← last cell, layer 85

Geological meaning:
Layers 1–35 (Tarbert): Relatively smooth porosity ~0.25–0.35, moderate fluvial sandstone
Layers 36–85 (Upper Ness): Highly variable, drops sharply to near-zero in shale interbeds.

Porosity is one of your input features alongside log(Kx). A cell with high φ holds more fluid but high permeability K controls how fast it flows. Together they govern the Darcy pressure field your model is learning to predict.

Note: The core equation the simulator solves for each cell is,
∂S/∂t  =  - (1/φ) × ∇·(k/μ × ∇p)

to previte the devision by 0 when φ=0 they replace it by φ = 1e-7.



- SPE10MODEL2_TOPS.INC — Depth of Each Layer Top:

13,200 = 60 × 220 (cells per layer)
85 layers × 2 ft spacing = depths from 12,000 ft to 12,168 ft.


The format N*value is Eclipse shorthand for "repeat value N times":
13200*12000   → 13,200 cells all at depth 12,000 ft  ← Layer 1 top
13200*12002   → 13,200 cells all at depth 12,002 ft  ← Layer 2 top
13200*12004   → ...                                  ← Layer 3 top
...
13200*12168   → 13,200 cells all at depth 12168 ft   ← Layer 85 top

probabily I'm not gonna use it in the FNO 


- SPE10MODEL2_PERM.INC — Permeability in All Three Directions:

Contains three sequential blocks, each with 1,122,000 values:


Block	Values	Meaning
PERMX	1,122,000	Permeability in x-direction (mD)
PERMY	1,122,000	Permeability in y-direction (mD)
PERMZ	1,122,000	Permeability in z-direction (mD, typically Kz ≈ 0.1 × Kx)

Total: 3,366,000 values across the whole file.

Key characteristics:
Range: near 0 mD (tight shale, no flow) to >10,000 mD (high-perm channels)
Spans ~5 orders of magnitude — this is why you use log(K) as FNO input, not raw K
Tarbert (layers 1–35): smoother, moderate values ~10–1000 mD
Upper Ness (layers 36–85): extreme contrasts, channels of > 5000 mD next to near-zero shale

- SPE10_MODEL2.DATA — The Full Simulation Deck:

SWOF table:

Sw (Water saturation)-> presentage of water to oil,(eg. 0.2 means 20% water 80% oil).
At the start of the simulation, every cell has Sw = 0.2 — meaning the reservoir begins nearly full of oil with just a thin film of irreducible water coating the rock grains. As water injection proceeds, Sw rises in each cell as water pushes oil out.
so it start with sw 0.2 and end with 0.8

Krw (Relative permeability of water)-> How easily water flows in presence of other fluids

Kro (Relative permeability of oil)-> How easily oil flows in presence of other fluids

as Sw water saturation increase the Krw ↑ (water flows more) and Kro ↓ (oil flows less)

-------------------------------------------------------------------------