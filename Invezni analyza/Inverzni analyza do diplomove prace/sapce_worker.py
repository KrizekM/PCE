
import sys, os, json, pickle
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import openturns as ot
ot.Log.Show(ot.Log.NONE)
from sapce import SensitivityAdaptivePCE

args     = json.loads(sys.argv[1])
in_file  = args['in_file']
out_file = args['out_file']
max_cond = args.get('max_cond', 1e2)
cr       = args.get('cr', 1e-8)

with open(in_file, 'rb') as f:
    data = pickle.load(f)

dist_joint = data['dist_joint']
Z_tr       = data['Z_tr']       # (n_train, 12)
Y_tr       = data['Y_tr']       # (n_train, 113)
Z_test     = data['Z_test']     # (n_test, 12)

n_outputs = Y_tr.shape[1]
n_test    = Z_test.shape[0]
Y_pred    = np.full((n_test, n_outputs), np.nan)

for i in range(n_outputs):
    try:
        # Trénuj SAPCE pro jeden výstup — vlastní adaptivní báze
        y_tr_i = Y_tr[:, i:i+1]          # (n_train, 1)

        sapce_i = SensitivityAdaptivePCE(
            dist_joint, Z_tr, y_tr_i,
            max_partial_degree=10
        )
        sapce_i.construct_adaptive_basis(
            max_condition_number=max_cond,
            termination_info=False
        )
        pce_i = sapce_i.construct_coefficient_pruned_pce(cr=cr)

        pred = np.array(pce_i.predict(Z_test))
        if pred.ndim == 2:
            pred = pred.ravel()
        Y_pred[:, i] = pred

    except Exception:
        # Tento výstup selhal — zůstane nan
        pass

with open(out_file, 'wb') as f:
    pickle.dump(Y_pred, f)
