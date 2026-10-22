#%% Init

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

import gc
from pygamry.dtaq import get_pstat, DtaqOcv, DtaqReadZ
from pyimpspec import Circuit, DataSet, FitResult, parse_data, fit_circuit, parse_cdc


def clean_sigma_average(sigmas, tol=1.0, preferred_keep=3, minimum_keep=2):
    """
    Selects a clean conductivity average from replicate sigma values.

    Rule:
    1. Prefer a cluster of preferred_keep values whose max-min <= tol.
    2. If that fails, accept a smaller cluster of minimum_keep values.
    3. If no pair agrees, return NaN and flag failure.
    """

    from itertools import combinations

    sigmas = np.array(sigmas, dtype=float)
    n = len(sigmas)

    if n == 0:
        return np.nan, [], [], "failed_no_data"

    for k in range(min(preferred_keep, n), minimum_keep - 1, -1):
        valid_clusters = []

        for idxs in combinations(range(n), k):
            vals = sigmas[list(idxs)]

            if np.max(vals) - np.min(vals) <= tol:
                valid_clusters.append(idxs)

        if valid_clusters:
            best = min(
                valid_clusters,
                key=lambda idxs: np.max(sigmas[list(idxs)]) - np.min(sigmas[list(idxs)])
            )

            keep_idx = list(best)
            reject_idx = [i for i in range(n) if i not in keep_idx]
            clean_avg = float(np.mean(sigmas[keep_idx]))

            if len(reject_idx) == 0:
                status = f"accepted_all_{n}"
            else:
                status = f"accepted_{len(keep_idx)}_of_{n}"

            return clean_avg, keep_idx, reject_idx, status

    return np.nan, [], list(range(n)), "failed_no_agreeing_cluster"


def eis_run(outputFolder, inputFolder):

    try:
        with open(inputFolder + '/UPDATES.json', 'r') as conf_file:
            conf = json.load(conf_file)

    except Exception as e:
        print(f'couldnt locate UPDATES: {e}')
        raise

    psycho = conf['psycho']

    data_path = outputFolder
    empty_verif_text_name = "outputend.txt"
    suffix = 'NIMS'

    ocp_duration = 25
    ocp_sample_period = 1

    eis_max_freq = 1.2e6
    eis_min_freq = 5e3
    eis_ppd = 15

    z_guess = 30
    AC_RMS = 0.01

    Kcell = 201.06347
    alpha = 0.968 * 0.625

    MIN_REPS = 3
    MAX_REPS = 6
    SIGMA_TOL = 1.0


    if not os.path.exists(data_path):
        os.makedirs(data_path)

    pstat = get_pstat()

    # Configure OCV
    ocv = DtaqOcv(
        write_mode='interval',
        write_interval=int(60 / ocp_sample_period)
    )

    # Configure EIS
    eis = DtaqReadZ(
        mode='pot',
        readzspeed='ReadZSpeedNorm',
        write_mode='continuous'
    )

    sigmas = []
    fit_Rs = []
    replicate_status = []

    for i in range(MAX_REPS):

        # Run OCV
        print(f'Running OCV - iter {i + 1}')
        ocv_file = os.path.join(data_path, f'OCV_{psycho}{suffix}.DTA')

        ocv.run(
            pstat,
            ocp_duration,
            ocp_sample_period,
            show_plot=True,
            result_file=ocv_file
        )

        print('OCV done\n')

        plt.close()
        time.sleep(1)

        # Get measured OCV for EIS
        V_oc = np.mean(ocv.dataframe['Vf'].values[-10:])
        V_dc = V_oc

        # Get frequencies to measure
        num_decades = np.log10(eis_max_freq) - np.log10(eis_min_freq)
        num_freq = int(eis_ppd * num_decades) + 1

        eis_freq = np.logspace(
            np.log10(eis_max_freq),
            np.log10(eis_min_freq),
            num_freq
        )

        # Run EIS
        print(f'Running EIS - iter {i + 1}')
        eis_file = os.path.join(data_path, f'EIS_{psycho}_{i}{suffix}.DTA')

        eis.run(
            pstat,
            eis_freq,
            V_dc,
            AC_RMS,
            z_guess,
            timeout=60,
            show_plot=True,
            plot_interval=1,
            plot_type='all',
            result_file=eis_file
        )

        print('EIS done\n')

        plt.close()
        time.sleep(5)

        # Parse EIS and fit circuit
        R = None
        sigma = None

        try:
            with open(eis_file, 'r'):
                data: DataSet = parse_data(eis_file)[0]

            circuit: Circuit = parse_cdc("R(Q[R(QR)])") #("RQ")
            fit: FitResult = fit_circuit(circuit, data)
            fit.circuit.to_sympy()

            df: DataFrame = fit.to_parameters_dataframe(running=True)
            # R = float(df.loc[df['Parameter'] == 'R', 'Value'].values[0])
            R = float(df.loc[(df['Element'] == 'R_0') & (df['Parameter'] == 'R'), 'Value'].iloc[0])
            
            print('new circuit shit:', df[['Element', 'Parameter', 'Value']])

            sigma = alpha * Kcell / R

            sigmas.append(sigma)
            fit_Rs.append(R)

            replicate_status.append({
                "replicate": i + 1,
                "fit_ok": True,
                "R": R,
                "sigma": sigma
            })

            print(f"Replicate {i + 1}: R = {R:.6f}, sigma = {sigma:.6f}")

        except Exception as e:
            print(f'Could not access/fit EIS result in iter {i + 1}: {e}')

            replicate_status.append({
                "replicate": i + 1,
                "fit_ok": False,
                "R": None,
                "sigma": None
            })

            continue

    # After at least 3 successful sigma values, check agreement
        if len(sigmas) >= MIN_REPS:

            temp_avg, temp_keep_idx, temp_reject_idx, temp_status = clean_sigma_average(
                sigmas,
                tol=SIGMA_TOL,
                preferred_keep=MIN_REPS,
                minimum_keep=2)

            if len(temp_keep_idx) >= MIN_REPS:
                print(f"Hell ya. {MIN_REPS} EIS replicates agree!")
                break

            elif i + 1 < MAX_REPS:
                print("Not enough agreeing EIS replicates. Running another...")
                continue

            else:
                print(f"Maximum EIS attempts reached. Temporary status: {temp_status}")
                break

    # Final average
    avg_sigma, keep_idx, reject_idx, sigma_status = clean_sigma_average(
        sigmas,
        tol=SIGMA_TOL,
        preferred_keep=MIN_REPS,
        minimum_keep=2
    )

    if np.isnan(avg_sigma):
        print("WARNING: No reliable sigma cluster found. Saving NaN to results.csv.")

    else:
        print(f"Final sigma average = {avg_sigma:.6f}")
        print(f"Accepted replicate indexes: {[i + 1 for i in keep_idx]}")
        print(f"Rejected replicate indexes: {[i + 1 for i in reject_idx]}")
        print(f"Sigma status: {sigma_status}")


    # Diagnostic EIS replicate output
    cat_save_path = os.path.join(data_path, 'result_EIS_cat.csv')

    diag_rows = []

    for idx, sigma_val in enumerate(sigmas):
        diag_rows.append({
            "psycho": psycho,
            "replicate": idx + 1,
            "sigma": sigma_val,
            "R": fit_Rs[idx] if idx < len(fit_Rs) else np.nan,
            "accepted": idx in keep_idx,
            "rejected": idx in reject_idx,
            "clean_avg_sigma": avg_sigma,
            "status": sigma_status,
            "tolerance": SIGMA_TOL,
            "n_successful_replicates": len(sigmas)
        })

    # Also store failed fitting attempts, if any happened
    for item in replicate_status:
        if not item["fit_ok"]:
            diag_rows.append({
                "psycho": psycho,
                "replicate": item["replicate"],
                "sigma": np.nan,
                "R": np.nan,
                "accepted": False,
                "rejected": True,
                "clean_avg_sigma": avg_sigma,
                "status": "fit_failed",
                "tolerance": SIGMA_TOL,
                "n_successful_replicates": len(sigmas)
            })

    diag_df = pd.DataFrame(diag_rows)

    write_header = not os.path.exists(cat_save_path)
    diag_df.to_csv(cat_save_path, mode='a', index=False, header=write_header)


    close_pstat = getattr(pstat, "Close", None)
    if callable(close_pstat):
        try:
            close_pstat()
            print("EIS potentiostat closed.")
        except Exception as exc:
            print(f"EIS potentiostat Close() failed: {exc}")
    else:
        print(f"EIS potentiostat has no Close() method: {type(pstat)}")
    
    del eis, ocv, pstat
    gc.collect()
    print("EIS potentiostat released.")
    
    return avg_sigma, sigma_status

