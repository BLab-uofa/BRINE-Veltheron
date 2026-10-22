# lsv_init.py
import subprocess, os, json, tempfile, sys, copy, csv
from typing import Optional, Dict, Any

DEFAULT_PY37 = r"C:\Program Files (x86)\Gamry Instruments\Python\Python37-32\python.exe"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MAIN = os.path.join(_THIS_DIR, "lsv_run.py")
_DEFAULT_OUT  = r"c:\Users\Blab2\Desktop\BRINE Veltheron\EXPOutput"

DEFAULT_PARAMS: Dict[str, Any] = {
    "lsv": {
        "reps": 1,
        "rest_secs": 2.0,
        "initial_voltage": 1.0,
        "apex1_voltage": 10.0,
        "apex2_voltage": 10.1,
        "final_voltage": 10.0, 
        "scan_rate": 0.02,
        "step_size": 0.01,
        "cycles": 1,
        "apex1_hold": 0.0,
        "apex2_hold": 0.0,
        "final_hold": 0.0,
        "max_points": 100000,
        "i_trip": [0.0001, 0.0001], #used to be 0.00009 # 0.0000163 # .0000179 0.0000008947364, [0.0003, 0.0004]
        "v_arm": 10.1,
        "use_abs_current": False,
        "lsv_mode": "BOTH"} #"HER/OER/BOTH"
}

def _params_for_side(params, side):
    p = copy.deepcopy(DEFAULT_PARAMS if params is None else params)

    side = str(side).upper()

    if side not in {"HER", "OER"}:
        raise ValueError("LSV side must be HER or OER.")

    lsv = p.get("lsv", p)
    lsv["lsv_mode"] = side

    return p


def _write_temp_params(params: Dict[str, Any]) -> str:
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    try:
        json.dump(params, tf)
        return tf.name
    finally:
        tf.close()


def lsv_run(
    params: Optional[Dict[str, Any]] = None,
    outdir: Optional[str] = None,
    py37: Optional[str] = None,
    main_path: Optional[str] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    side = None,) -> str:
    
    
    if side is None:
        params = copy.deepcopy(DEFAULT_PARAMS if params is None else params)

        mode = str(params.get("lsv", params).get("lsv_mode", "OER")).upper()

        if mode == "BOTH":
            raise ValueError("lsv_run() cannot execute BOTH in one subprocess anymore. "
                             "Call once with side='HER' and once with side='OER'.")

    else:
        params = _params_for_side(params, side)
    
    py37 = py37 or DEFAULT_PY37
    main_path = main_path or _DEFAULT_MAIN
    outdir = outdir or _DEFAULT_OUT
    os.makedirs(outdir, exist_ok=True)
    params_path = _write_temp_params(params)
    cmd = [py37, "-u", main_path, "--params", params_path, "--out-dir", outdir]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            cwd=cwd or _THIS_DIR,
            timeout=timeout,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        e.stdout = e.output
        raise
    finally:
        try:
            os.unlink(params_path)
        except Exception:
            pass


def start_lsv_async(
    params: Optional[Dict[str, Any]] = None,
    outdir: Optional[str] = None,
    py37: Optional[str] = None,
    main_path: Optional[str] = None,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    params = DEFAULT_PARAMS if params is None else params
    py37 = py37 or DEFAULT_PY37
    main_path = main_path or _DEFAULT_MAIN
    outdir = outdir or _DEFAULT_OUT
    os.makedirs(outdir, exist_ok=True)
    params_path = _write_temp_params(params)
    cmd = [py37, "-u", main_path, "--params", params_path, "--out-dir", outdir]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd or _THIS_DIR,
    )
    proc._lsv_params_path = params_path  # type: ignore[attr-defined]
    return proc



def _read_endpoint(outdir, psycho, side):
    path = os.path.join(outdir, f"lsv_endpoint_{psycho}_{side}.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LSV endpoint not found: {path}"
        )

    with open(path, "r") as f:
        data = json.load(f)

    return float(data["voltage"])


def finalize_lsv_result(outdir, psycho, lsv_mode, conductivity = None):
    mode = str(lsv_mode).upper()

    if mode not in {"HER", "OER", "BOTH"}:
        raise ValueError(
            "lsv_mode must be HER, OER, or BOTH."
        )

    V_HER = None
    V_OER = None

    if mode in {"HER", "BOTH"}:
        V_HER = _read_endpoint(
            outdir,
            psycho,
            "HER"
        )

    if mode in {"OER", "BOTH"}:
        V_OER = _read_endpoint(
            outdir,
            psycho,
            "OER"
        )


    if mode == "BOTH":
        ESW = V_OER - V_HER

    elif mode == "OER":
        ESW = V_OER

    else:
        ESW = -V_HER


    # ---------------------------------------------
    # Detailed LSV record
    # ---------------------------------------------

    cat_path = os.path.join(
        outdir,
        "result_LSV_cat.csv"
    )

    cols = [
        "psycho",
        "lsv_mode",
        "V_OER",
        "V_HER",
        "ESW"
    ]

    write_header = not os.path.exists(cat_path)

    with open(cat_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=cols
        )

        if write_header:
            writer.writeheader()

        writer.writerow({
            "psycho": psycho,
            "lsv_mode": mode,
            "V_OER": f"{V_OER:.6f}" if V_OER is not None else "",
            "V_HER": f"{V_HER:.6f}" if V_HER is not None else "",
            "ESW": f"{ESW:.6f}"
        })


    # ---------------------------------------------
    # Normal BRINE objective output
    # ---------------------------------------------

    results_path = os.path.join(
        outdir,
        "results.csv"
    )


    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)

        if mode == "BOTH":
            HER_score = -V_HER
            OER_score = V_OER
            
            if conductivity is not None:
                writer.writerow([f"{conductivity:.6f}", f"{HER_score:.6f}", f"{OER_score:.6f}"])
            else:
                writer.writerow([f"{HER_score:.6f}", f"{OER_score:.6f}"])
            

        elif mode == "HER":
            writer.writerow([f"{-V_HER:.6f}"])

        elif mode == "OER":
            writer.writerow([f"{V_OER:.6f}"])




    print(
        f"LSV finalized: mode={mode}, "
        f"V_HER={V_HER}, "
        f"V_OER={V_OER}, "
        f"ESW={ESW:.6f} V"
    )

    return ESW


def cleanup_async(proc: subprocess.Popen) -> None:
    try:
        params_path = getattr(proc, "_lsv_params_path", None)
        if params_path and os.path.exists(params_path):
            os.unlink(params_path)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        print(lsv_run())
    except subprocess.CalledProcessError as e:
        print("----- begin child output -----", file=sys.stderr)
        print(e.stdout or "", file=sys.stderr)
        print("----- end child output -----", file=sys.stderr)
        sys.exit(e.returncode)