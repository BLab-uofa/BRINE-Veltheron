#!/usr/bin/env python
# lsv_run.py
# Runs LSV only and writes:
#   EXPOutput/results.csv         (one value: VSW)
#   EXPOutput/outputened.txt and outputend.txt as success flags.

import os, time, json, argparse, sys, csv
import numpy as np

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt

import toolkitpy as tkp


def _append_row(path, row_1d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "ab") as f:
        np.savetxt(f, [row_1d], delimiter=",", fmt="%.6f", comments="")


def _read_updates_psycho():
    try:
        with open('./UPDATES.json', 'r') as conf_file:
            conf = json.load(conf_file)
        return conf.get('psycho', 'X')
    except Exception:
        return 'X'


def _endpoint_path(out_dir, psycho, side):
    return os.path.join(out_dir, f"lsv_endpoint_{psycho}_{side}.json")


def _write_endpoint(out_dir, psycho, side, voltage):
    path = _endpoint_path(out_dir, psycho, side)

    payload = {
        "psycho": psycho,
        "side": side,
        "voltage": float(voltage)
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[LSV] Endpoint written: {path}")


def _resolve_i_trip(i_trip_values, side):
    if isinstance(i_trip_values, (list, tuple)):
        if len(i_trip_values) == 0:
            raise ValueError("i_trip list is empty.")

        if side == "HER":
            return abs(float(i_trip_values[0]))

        if len(i_trip_values) >= 2:
            return abs(float(i_trip_values[1]))

        return abs(float(i_trip_values[0]))

    return abs(float(i_trip_values))

def _make_lsv_direction_params(lsv_params, side):
    p = dict(lsv_params)

    ss = float(p.get("step_size", 0.01))
    arm = max(0.10, 5 * ss)

    fv_abs = abs(float(p.get("final_voltage", 10.0)))
    iv = float(p.get("initial_voltage", 0.0))

    # p["initial_voltage"] = 0.0
    p["one_way"] = True
    p["use_abs_current"] = False


    if side == "OER":
        p["final_voltage"] = fv_abs
        p["trip_direction"] = "positive"
        p["v_arm"] = iv + arm

    elif side == "HER":
        p["final_voltage"] = -fv_abs
        p["trip_direction"] = "negative"
        p["v_arm"] = iv - arm

    else:
        raise ValueError(f"Unknown LSV side: {side}")

    return p


def _append_lsv_cat(out_dir, row):
    path = os.path.join(out_dir, "result_LSV_cat.csv")

    cols = [
        "psycho",
        "lsv_mode",
        "V_OER",
        "V_HER",
        "ESW"
    ]

    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def _safe_set_title(ax, title):
    try:
        ax.set_title(title)
    except Exception:
        pass


def run_lsv_for_vsw(lsv_params, out_dir, psycho, lsv_label=None):
    def _hard_stop(pstat, curve):
        try:
            if hasattr(pstat, "set_cell"):
                pstat.set_cell(False)
        except Exception:
            pass
        for meth in ("stop", "stop_signal", "abort_signal", "abort", "halt"):
            try:
                if hasattr(pstat, meth):
                    getattr(pstat, meth)()
            except Exception:
                pass
        try:
            if hasattr(curve, "stop"):
                curve.stop()
        except Exception:
            pass

    os.makedirs(out_dir, exist_ok=True)

    reps   = int(lsv_params.get("reps", 1))
    rest_s = float(lsv_params.get("rest_secs", 2.0))

    iv = float(lsv_params["initial_voltage"])
    a1 = float(lsv_params["apex1_voltage"])
    a2 = float(lsv_params["apex2_voltage"])
    fv = float(lsv_params["final_voltage"])
    sr = float(lsv_params["scan_rate"])
    ss = float(lsv_params["step_size"])
    cyc = int(lsv_params.get("cycles", 1))
    h1 = float(lsv_params.get("apex1_hold", 0.0))
    h2 = float(lsv_params.get("apex2_hold", 0.0))
    hf = float(lsv_params.get("final_hold", 0.0))
    max_pts = int(lsv_params.get("max_points", 20000))

    one_way = bool(lsv_params.get("one_way", True))
    if one_way:
        a1 = fv
        a2 = fv - max(ss, 1e-3)
        h1 = float(lsv_params.get("apex1_hold", 0.0) or 0.0)
        h2 = float(lsv_params.get("apex2_hold", 0.0) or 0.0)
        hf = float(lsv_params.get("final_hold", 0.0) or 0.0)

    print(f"[LSV] Params: iv={iv}, a1={a1}, a2={a2}, fv={fv}, sr={sr}, ss={ss}, cyc={cyc}")


    # I_TRIP = float(lsv_params.get("i_trip", 0.0001))
    # use_abs_current = bool(lsv_params.get("use_abs_current", False))
    # I need 2-sided LSV now for both HER and OER Capture
    I_TRIP = float(lsv_params.get("i_trip", 0.0001))
    use_abs_current = bool(lsv_params.get("use_abs_current", False))
    trip_direction = str(lsv_params.get("trip_direction","positive")).lower()

    # Critical fix:
    # Do not allow trip at startup before the sweep has actually progressed.
    v_arm = lsv_params.get("v_arm", None)
    if v_arm is None:
        if fv >= iv:
            V_ARM = iv + max(0.10, 5 * ss)
        else:
            V_ARM = iv - max(0.10, 5 * ss)
    else:
        V_ARM = float(v_arm)

    # If user gave an impossible arm voltage, clamp it into the sweep window.
    if fv >= iv and V_ARM >= fv:
        V_ARM = iv + max(0.10, 5 * ss)
    elif fv < iv and V_ARM <= fv:
        V_ARM = iv - max(0.10, 5 * ss)

    print(f"[LSV] Trip settings: I_TRIP={I_TRIP}, V_ARM={V_ARM}, use_abs_current={use_abs_current}, trip_direction={trip_direction}")

    session = f"lsv_run_{os.getpid()}"
    tkp.toolkitpy_init(session)

    for attempt in range(1, 5):
        sections = list(tkp.enum_sections())
        print(f"[LSV] Device attempt {attempt}/4; sections={sections}", flush=True)

        try:
            if not sections:
                raise RuntimeError("ToolkitPy found no potentiostat sections.")
            pstat = tkp.Pstat(sections[0])
            print("[LSV] Potentiostat opened.", flush=True)
            break
        except Exception as exc:
            if "DEVICE_NOTFOUND" not in type(exc).__name__ or attempt == 4:
                raise
            print(f"[LSV] Device not found; retrying in 3 s: {exc}", flush=True)
            time.sleep(3)

    pstat.set_ctrl_mode(tkp.PSTATMODE)

    # Keep this gentle and binding-safe.
    target_v = float(lsv_params.get("final_voltage", 5))
    headroom = max(5.0, abs(target_v) + 0.3)

    # IMPORTANT: vch_range_value_list() returns range IDs, not voltage values.
    # Force the widest Vch measurement range using an INTEGER ID. Casting the
    # ID to float (for example 3.0) can be interpreted as a ~3 V range and can
    # produce the repeatable ~3.28 V measured-voltage ceiling.
    vch_range_configured = False
    if hasattr(pstat, "vch_range_value_list") and hasattr(pstat, "set_vch_range"):
        try:
            vals = list(pstat.vch_range_value_list())
            labels = None
            if hasattr(pstat, "vch_range_label_list"):
                try:
                    labels = list(pstat.vch_range_label_list())
                except Exception:
                    labels = None

            print(f"[LSV] Available Vch range IDs: {vals}")
            if labels:
                print(f"[LSV] Available Vch range labels: {labels}")

            if not vals:
                raise RuntimeError("No Vch ranges were returned by ToolkitPy.")

            widest_vch_id = int(max(vals))
            pstat.set_vch_range(widest_vch_id)
            vch_range_configured = True
            print(f"[LSV] Forced widest Vch range ID: {widest_vch_id} (integer)")
        except Exception as e:
            print(f"[LSV] widest Vch range setup failed: {e}")

    # I/E autorange controls CURRENT measurement only. Keep it separate from Vch.
    ie_autorange_enabled = False
    for meth in ("set_ie_range_mode", "set_ie_range_mode_auto"):
        if hasattr(pstat, meth):
            try:
                getattr(pstat, meth)(True)
                ie_autorange_enabled = True
                print(f"[LSV] I/E current autorange enabled via {meth}(True)")
                break
            except Exception as e:
                print(f"[LSV] {meth}(True) failed: {e}")

    # If a fixed wide Vch range could not be selected, enable Vch autorange
    # independently as the fallback. The old combined loop stopped after
    # enabling I/E autorange, so the Vch call was never reached.
    vch_autorange_enabled = False
    if not vch_range_configured:
        for meth in ("set_vch_range_mode", "set_vch_range_mode_auto"):
            if hasattr(pstat, meth):
                try:
                    getattr(pstat, meth)(True)
                    vch_autorange_enabled = True
                    print(f"[LSV] Vch autorange enabled via {meth}(True)")
                    break
                except Exception as e:
                    print(f"[LSV] {meth}(True) failed: {e}")

    if abs(target_v) > 3.2 and not (vch_range_configured or vch_autorange_enabled):
        raise RuntimeError(
            "Target exceeds 3.2 V, but neither the wide Vch range nor Vch "
            "autorange could be configured. Aborting before energizing the cell."
        )

    # set_scan_range: your binding wants int
    if hasattr(pstat, "set_scan_range"):
        try:
            if hasattr(pstat, "scan_range_label_list"):
                labels = pstat.scan_range_label_list()
                if labels:
                    # widest range by index
                    pstat.set_scan_range(int(len(labels) - 1))
                    print(f"[LSV] set_scan_range({int(len(labels) - 1)})")
            else:
                pstat.set_scan_range(3)
                print("[LSV] set_scan_range(3)")
        except Exception as e:
            print(f"[LSV] scan_range bump failed: {e}")

    # set_compliance_voltage may not exist on your device/binding
    if hasattr(pstat, "set_compliance_voltage"):
        try:
            pstat.set_compliance_voltage(headroom)
            print(f"[LSV] set_compliance_voltage({headroom})")
        except Exception as e:
            print(f"[LSV] set_compliance_voltage failed: {e}")

    # Correct signature: enable, max_value, min_value
    if hasattr(pstat, "set_safe_cell_voltage_limits"):
        try:
            pstat.set_safe_cell_voltage_limits(True, headroom, -headroom)
            print(f"[LSV] set_safe_cell_voltage_limits(True, {headroom}, {-headroom})")
        except Exception as e:
            print(f"[LSV] set_safe_cell_voltage_limits failed: {e}")

    plt.ion()
    fig, ax = plt.subplots()
    ln, = ax.plot([], [], marker=".", markersize=3, linestyle="-", linewidth=1.0)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    try:
        fig.canvas.manager.set_window_title("LSV live")
    except Exception:
        pass
    try:
        fig.show()
    except Exception:
        pass

    VSW = float("nan")

    for r in range(reps):
        dt = ss / sr
        if dt <= 0:
            raise ValueError("Computed data interval dt <= 0 (check step_size/scan_rate).")

        curve = tkp.RcvCurve(pstat, max_pts)
        sig = pstat.signal_r_up_dn_new(
            [iv, a1, a2, fv],
            [sr, sr, sr],
            [h1, h2, hf],
            dt,
            cyc,
            tkp.PSTATMODE
        )
        pstat.set_signal_r_up_dn(sig)
        pstat.init_signal()
        pstat.set_cell(True)
        curve.run(True)

        all_T, all_Vf, all_Im, all_Cy = [], [], [], []
        stopped = False
        stop_v  = float("nan")

        sweep_secs = (abs(iv - a1) + abs(a1 - a2) + abs(a2 - fv)) / sr
        hold_secs  = (h1 + h2 + hf)
        expected_secs = (sweep_secs + hold_secs) * max(1, cyc)
        timeout_secs  = expected_secs + 20.0
        start_time = time.time()
        last_data  = start_time
        last_recorded_time = float("-inf")
        recent_v_points = []
        recent_t_points = []
        clamp_detected = False
        printed_fields = False

        while True:
            if (time.time() - start_time) > timeout_secs:
                print(f"[LSV] Watchdog timeout ({timeout_secs:.1f}s) forcing stop.")
                _hard_stop(pstat, curve)
                break

            if not curve.running():
                break

            try:
                d = curve.acq_data()
                if d.size:
                    names = {n.lower(): n for n in d.dtype.names}
                    if not printed_fields:
                        print(f"[LSV] Acquisition fields: {d.dtype.names}")
                        printed_fields = True

                    T_all  = d[names["time"]]
                    Vf_all = d[names["vf"]]
                    Im_all = d[names["im"]]
                    Cy_all = d[names["cycle"]] if "cycle" in names else np.zeros_like(Vf_all)

                    # RcvCurve.acq_data() can return cumulative data. Keep only
                    # genuinely new time points so the CSV and clamp detector are
                    # not flooded with duplicate history.
                    new_mask = T_all > (last_recorded_time + 1e-12)
                    if not np.any(new_mask):
                        time.sleep(0.02)
                        continue

                    T  = T_all[new_mask]
                    Vf = Vf_all[new_mask]
                    Im = Im_all[new_mask]
                    Cy = Cy_all[new_mask]
                    last_recorded_time = float(T[-1])

                    # Icheck = np.abs(Im) if use_abs_current else Im #NO NEED. REMOVE SOON

                    if fv >= iv:
                        armed = (Vf >= V_ARM)
                        EPS = max(ss, 1e-6)
                        mask_vmax = (Vf >= (fv - EPS))
                    else:
                        armed = (Vf <= V_ARM)
                        EPS = max(ss, 1e-6)
                        mask_vmax = (Vf <= (fv + EPS))

                    # mask_trip = armed & (Icheck > I_TRIP) # SO I HAVE TO CHYANGE THIS
                    if use_abs_current:
                        mask_trip = armed & (np.abs(Im) > I_TRIP)
                    elif trip_direction == "negative":
                        mask_trip = armed & (Im < -I_TRIP)
                    else:
                        mask_trip = armed & (Im > I_TRIP)

                    if np.any(mask_trip) or np.any(mask_vmax):
                        idx = int(np.argmax(mask_trip | mask_vmax))
                        T, Vf, Im, Cy = T[:idx+1], Vf[:idx+1], Im[:idx+1], Cy[:idx+1]
                        stop_v = float(Vf[-1])
                        stopped = True

                    all_T.append(T)
                    all_Vf.append(Vf)
                    all_Im.append(Im)
                    all_Cy.append(Cy)

                    # Keep a short window of unique recent points. If the signal
                    # should still be ramping but measured Vf becomes exactly flat
                    # far below the target, stop before the electrode is exposed to
                    # a much larger hidden command voltage.
                    recent_v_points.extend(np.asarray(Vf, dtype=float).tolist())
                    recent_t_points.extend(np.asarray(T, dtype=float).tolist())
                    recent_v_points = recent_v_points[-30:]
                    recent_t_points = recent_t_points[-30:]

                    if len(recent_v_points) >= 20 and not stopped:
                        recent_v = np.asarray(recent_v_points, dtype=float)
                        recent_t = np.asarray(recent_t_points, dtype=float)
                        v_span = float(np.nanmax(recent_v) - np.nanmin(recent_v))
                        elapsed_window = float(recent_t[-1] - recent_t[0])
                        expected_ramp = abs(sr) * elapsed_window
                        v_mean = float(np.nanmean(recent_v))

                        commanded_far_away = abs(fv - v_mean) > 0.5
                        should_be_moving = expected_ramp > max(5 * ss, 0.05)
                        nearly_flat = v_span < max(2 * ss, 0.02)

                        if commanded_far_away and should_be_moving and nearly_flat:
                            clamp_detected = True
                            stop_v = float("nan")

                            last = d[-1]
                            print(
                                f"[LSV] Clamp diagnostics: time={float(last[names['time']]):.2f} s, "
                                f"Vf={float(last[names['vf']]):.6f} V, "
                                f"Vsig={float(last[names['vsig']]):.6f} V, "
                                f"Im={float(last[names['im']]):.6e} A, "
                                f"overload={last[names['overload']]}, "
                                f"stop_test={last[names['stop_test']]}")


                            _hard_stop(pstat, curve)
                            break

                    try:
                        # Plot the complete accumulated scan, not only the newest
                        # acquisition chunk. After duplicate filtering, a chunk is
                        # often just one point, which is invisible without a marker.
                        plot_v = np.concatenate(all_Vf)
                        plot_i = np.concatenate(all_Im)
                        ln.set_data(plot_v, plot_i)
                        _safe_set_title(ax, f"LSV live (rep {r+1}/{reps})")
                        ax.relim()
                        ax.autoscale_view()
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        plt.pause(0.05)
                    except Exception as e:
                        print(f"[LSV] live plot warning: {e}")

                    last_data = time.time()

                    if stopped:
                        _hard_stop(pstat, curve)
                        break
                else:
                    if (time.time() - last_data) > max(5 * (ss / sr), 2.0):
                        print("[LSV] No data received for a while forcing stop.")
                        _hard_stop(pstat, curve)
                        break
            except Exception as e:
                print(f"[LSV] acquisition warning: {e}")

            time.sleep(0.1)

        if not stopped:
            _hard_stop(pstat, curve)

        if all_T:
            T  = np.concatenate(all_T)
            Vf = np.concatenate(all_Vf)
            Im = np.concatenate(all_Im)
            Cy = np.concatenate(all_Cy)
            arr = np.column_stack((T, Vf, Im, Cy))
        else:
            arr = np.empty((0, 4))

        # csv_name = f"lsv_{psycho}_r{r+1}.csv" if reps > 1 else f"lsv_{psycho}.csv" #HERE TOO
        label_suffix = f"_{lsv_label}" if lsv_label else ""
        csv_name = f"lsv_{psycho}{label_suffix}_r{r+1}.csv" if reps > 1 else f"lsv_{psycho}{label_suffix}.csv"
        
        
        csv_path = os.path.join(out_dir, csv_name)
        np.savetxt(csv_path, arr, fmt="%.6e", delimiter=",", header="Time,Vf,Im,Cycle", comments="")
        print("LSV CSV written:", csv_path)

        if clamp_detected:
            raise RuntimeError(
                "Measured-voltage clamp detected. The scan was rejected and no "
                "OER/HER endpoint should be reported from this run."
            )

        if stopped:
            
            if use_abs_current:
                pol = f"abs(Im) > {I_TRIP}"
            elif trip_direction == "negative":
                pol = f"Im < -{I_TRIP}"
            else:
                pol = f"Im > {I_TRIP}"
            print(f"*** LSV stopped at Vf = {stop_v:.6f} V ({pol} A after arming)...")   
            
        else:
            if arr.shape[0] > 0:
                stop_v = float(arr[-1, 1])
                print(f"LSV completed without trip; final Vf = {stop_v:.6f} V...")
            else:
                stop_v = float("nan")
                print("LSV captured no data.")

        VSW = stop_v

        if r < reps - 1:
            time.sleep(rest_s)

    plt.ioff()
    try:
        plt.close(fig)
    except Exception:
        pass

    return VSW


def main():
    p = argparse.ArgumentParser(description="Run LSV only; save VSW to results.csv and flag files")
    p.add_argument("--params", required=True, help="JSON string or path to JSON file with LSV parameters")
    p.add_argument("--out-dir", required=True, help="Directory to write outputs")
    args = p.parse_args()

    if os.path.isfile(args.params):
        with open(args.params, "r") as f:
            params = json.load(f)
    else:
        params = json.loads(args.params)

    lsv_params = params.get("lsv", params)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    psycho = _read_updates_psycho()

    lsv_mode = str(lsv_params.get("lsv_mode", "OER")).upper()

    if lsv_mode not in {"HER", "OER"}:
        raise ValueError(
            "lsv_run.py now executes exactly one LSV direction per process. "
            "Use HER or OER here. BOTH is orchestrated outside this Python 3.7 process."
        )

    i_trip_values = lsv_params["i_trip"]

    direction_params = _make_lsv_direction_params(lsv_params, lsv_mode)
    direction_params["i_trip"] = _resolve_i_trip(i_trip_values, lsv_mode)


    # Delete any endpoint from a previous failed/repeated attempt
    # at this same experimental index.
    endpoint_file = _endpoint_path(out_dir, psycho, lsv_mode)

    if os.path.exists(endpoint_file):
        os.remove(endpoint_file)


    V_stop = run_lsv_for_vsw(
        direction_params,
        out_dir,
        psycho,
        lsv_label=lsv_mode
    )

    print(f"{lsv_mode} stop voltage: {V_stop:.6f} V")

    _write_endpoint(
        out_dir,
        psycho,
        lsv_mode,
        V_stop
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)