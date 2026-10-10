import warnings; warnings.filterwarnings('ignore')
import sys, gc, dataclasses as dc, numpy as np, do_analysis as da
m = da.laad_volledig('model_full_tuned.pkl')
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'vorm_dag', 'diepte_dag'): setattr(m, a, None)
gc.collect()
aan = sys.argv[1] == 'aan'; dag = 'dag' in sys.argv
m.params = dc.replace(m.params, piek_diepte=aan)
t, _, _, r = da.logloss_eerlijk(m, dag=dag, per_rij=True)
r.astype(np.float32).to_pickle(f'cmp_{sys.argv[1]}_{"dag" if dag else "vooraf"}.pkl'); print(sys.argv[1], 'dag' if dag else 'vooraf', round(t, 5), flush=True)
print('e0', getattr(m, 'diepte_e0', None))
