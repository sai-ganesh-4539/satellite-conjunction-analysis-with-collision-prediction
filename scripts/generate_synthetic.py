"""Generate synthetic CDM data matching the real CSV structure."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
n = 2708

n_pairs = 50
pairs = [(25544 + i, 25544 + 100 + i) for i in range(n_pairs)]
pair_indices = np.random.choice(n_pairs, n)

cdm_id = list(range(1, n + 1))
sat1_ids, sat2_ids, tca_list, created_list = [], [], [], []
min_rngs, pcs = [], []
sat1_rcs, sat2_rcs, sat1_excl, sat2_excl = [], [], [], []
emergency, sat1_types, sat2_types = [], [], []
sat1_names, sat2_names = [], []
obj_types = ['PAY', 'R/B', 'DEB']
base_time = datetime(2024, 1, 1)

for i in range(n):
    p = pairs[pair_indices[i]]
    sat1_ids.append(p[0]); sat2_ids.append(p[1])
    sat1_names.append(f'SAT-{p[0]}'); sat2_names.append(f'SAT-{p[1]}')
    tca = base_time + timedelta(days=np.random.uniform(0, 180), hours=np.random.uniform(0, 24))
    tca_list.append(tca)
    created_list.append(tca - timedelta(hours=np.random.uniform(1, 72)))
    if np.random.random() < 0.05:
        min_rngs.append(np.random.exponential(0.05))
        pcs.append(np.random.uniform(1e-4, 1e-2))
        emergency.append(1)
    else:
        min_rngs.append(np.random.exponential(2.0) + 0.1)
        pcs.append(np.random.uniform(1e-8, 1e-5))
        emergency.append(0)
    sat1_rcs.append(np.random.choice([0.01, 0.1, 1.0, 5.0, 10.0]))
    sat2_rcs.append(np.random.choice([0.01, 0.1, 1.0, 5.0, 10.0]))
    sat1_excl.append(np.random.uniform(5, 50))
    sat2_excl.append(np.random.uniform(5, 50))
    sat1_types.append(np.random.choice(obj_types))
    sat2_types.append(np.random.choice(obj_types))

df = pd.DataFrame({
    'CDM_ID': cdm_id, 'CREATED': created_list, 'EMERGENCY_REPORTABLE': emergency,
    'TCA': tca_list, 'MIN_RNG': min_rngs, 'PC': pcs,
    'SAT_1_ID': sat1_ids, 'SAT_1_NAME': sat1_names, 'SAT1_OBJECT_TYPE': sat1_types,
    'SAT1_RCS': sat1_rcs, 'SAT_1_EXCL_VOL': sat1_excl,
    'SAT_2_ID': sat2_ids, 'SAT_2_NAME': sat2_names, 'SAT2_OBJECT_TYPE': sat2_types,
    'SAT2_RCS': sat2_rcs, 'SAT_2_EXCL_VOL': sat2_excl,
})
df = df.sort_values('TCA').reset_index(drop=True)
df.to_csv('data/cdms.csv', index=False)
print(f'Generated {len(df)} CDMs')
print(f'High-risk: {(df["PC"] > 1e-4).sum()} / {len(df)} ({100*(df["PC"] > 1e-4).mean():.1f}%)')
