#%% Init

import sys
import os
import json
import time
import uuid
import asyncio
import traceback
import websockets
import math

from subprocess import CalledProcessError
from websockets.exceptions import WebSocketException, ConnectionClosedError

import importlib
import lsv_initt
importlib.reload(lsv_initt)

from cv_init import cv_run
from eis_init import eis_run

from xarm.wrapper import XArmAPI


#%% Test sequence parsing

TEST_NAME_TO_CODE = {"E": "E", "EIS": "E", "C": "C", "CV": "C", "L": "L", "LSV": "L"}
VALID_SCENARIOS = {"E", "C", "L", "EC", "EL", "CL", "ECL"}

TEST_CODE_TO_NAME = {"E": "EIS", "C": "CV", "L": "LSV"}

WASH_AFTER_EIS = False
WASH_AFTER_CV = True
WASH_AFTER_LSV = True


#%% XArm / electrode settings

XARM_IP = "192.168.1.236"

SEAT_UP_MM = 1.5
SEAT_YAW = -3
TIP_DOWN_PITCH_DEG = -1.0
GRIP_POSITION = 60

OT2_ELECTRODES = {
    21: {"x": 44.5, "y": 22.5, "hover_z": 110.0, "pick_z": 74.0},
    22: {"x": 45.2, "y": -20.0, "hover_z": 110.0, "pick_z": 74.0}
}

XARM_ELECTRODES = {
    21: {
        "grab": [733.0, 72, 241.0, 0.0, -94.0, 178.9],
        "seat": [733.0, 72, 208.0, 0.0, -94.0, 178.9],
        "home": [573.0, 72, 241.0, 0.0, -94.0, 178.9]
    },

    22: {
        "grab": [733.5, 30, 241.0, 0.0, -94.0, 178.9],
        "seat": [733.5, 30, 208.0, 0.0, -94.0, 178.9],
        "home": [572.5, 30, 241.0, 0.0, -94.0, 178.9]
    }
}


def normalize_tests(tests):
    """
    Converts tests into a compact sequence string.

    Examples:
        ['EIS', 'CV', 'LSV'] -> 'ECL'
        ['CV', 'LSV']        -> 'CL'
        'ECL'                -> 'ECL'
    """

    if isinstance(tests, str):
        sequence = tests.upper().replace(" ", "").replace(",", "")

    elif isinstance(tests, list):
        sequence = ""

        for test in tests:
            key = str(test).upper().strip()

            if key not in TEST_NAME_TO_CODE:
                raise ValueError(f"Unknown test name: {test}")

            sequence += TEST_NAME_TO_CODE[key]

    else:
        raise TypeError("tests must be a string or a list.")

    if sequence not in VALID_SCENARIOS:
        raise ValueError(
            f"Invalid test sequence: {sequence}. "
            f"Allowed scenarios are: {sorted(VALID_SCENARIOS)}"
        )

    return sequence


#%% Robust WebSocket function

async def send_message_via_ws(ws_url, code, message_name):
    try:
        print(f"Starting WebSocket execution for {message_name}.")

        async with websockets.connect(ws_url, open_timeout=60) as websocket:
            msg_id = str(uuid.uuid4())

            msg = {
                "header": {
                    "msg_id": msg_id,
                    "username": "Reza",
                    "session": str(uuid.uuid4()),
                    "msg_type": "execute_request",
                    "version": "5.3"
                },
                "parent_header": {},
                "metadata": {},
                "content": {
                    "code": code,
                    "silent": False,
                    "store_history": True,
                    "user_expressions": {},
                    "allow_stdin": False
                }
            }

            await websocket.send(json.dumps(msg))
            print(f"Message sent for {message_name}. Waiting for response.")

            while True:
                raw = await websocket.recv()
                response = json.loads(raw)

                parent = response.get("parent_header", {})
                if parent.get("msg_id") != msg_id:
                    continue

                print(f"{message_name} response: {raw}")

                msg_type = response.get("msg_type")

                if msg_type == "error":
                    print("===== JUPYTER EXECUTION ERROR =====")
                    content = response.get("content", {})

                    print("ename :", content.get("ename"))
                    print("evalue:", content.get("evalue"))

                    tb = content.get("traceback", [])
                    for line in tb:
                        print(line)

                    raise RuntimeError(
                        f"{content.get('ename')}: {content.get('evalue')}"
                    )

                if msg_type == "stream":
                    text = response.get("content", {}).get("text", "")
                    if text:
                        print(text, end="")

                if msg_type == "execute_result":
                    print(response.get("content", {}))

                if msg_type == "execute_reply":
                    status = response.get("content", {}).get("status")

                    if status == "ok":
                        print(f"{message_name} execution completed successfully.")
                        return True

                    raise RuntimeError(f"Execution finished with status={status}")

    except Exception as e:
        print(f"WebSocket/general error during {message_name}: {repr(e)}")
        traceback.print_exc()
        return False


#%% Unified wash

def run_wash(ws_url):
    continuation_code = """
reservoir_c = res.reservoir_c
wash_h = 88.5

## WATER FIRST
############################################# [A6] prohibited for waste cell

tcW11 = reservoir_c['A10']
tcW12 = reservoir_c['A9']
tcW1 = tcW11 if cycle_no < 150 else tcW12

## ACID
tcA1 = reservoir_c['A7']

## WATER SECOND
tcW21 = reservoir_c['A1']
tcW22 = reservoir_c['A4']
tcW2 = tcW21 if cycle_no < 150 else tcW22

tcAW = [tcA1, tcW2]

tcE = reservoir_c['A12']

Rpipette.move_to(target_well.bottom(180), speed=30)

for _ in range(2):
    Rpipette.move_to(tcW1.bottom(z=wash_h).move(Point(y=+18)))
    protocol.delay(0.5)
    Rpipette.move_to(tcW1.bottom(z=wash_h).move(Point(y=-18)))
    protocol.delay(2)
Rpipette.move_to(tcW1.bottom(z=110))

for cnt in range(len(tcAW)):
    for _ in range(3):
        Rpipette.move_to(tcAW[cnt].bottom(z=wash_h).move(Point(y=+18)), speed=60)
        protocol.delay(0.5)
        Rpipette.move_to(tcAW[cnt].bottom(z=wash_h).move(Point(y=-18)), speed=60)
        protocol.delay(2) if tcAW[cnt] == tcA1 else protocol.delay(3)
    Rpipette.move_to(tcAW[cnt].bottom(z=110))

for _ in range(2):
    Rpipette.move_to(tcE.bottom(z=wash_h).move(Point(y=+12)), speed=80)
    protocol.delay(0.1)
    Rpipette.move_to(tcE.bottom(z=wash_h).move(Point(y=-12)), speed=80)
    protocol.delay(0.1)

# No drying location anymore. Leave the washed electrode safely raised here.
Rpipette.move_to(tcE.bottom(z=110))
"""

    print("Starting WebSocket: Acid washing alien tissue.")

    ok = asyncio.get_event_loop().run_until_complete(
        send_message_via_ws(ws_url, continuation_code, "continuation_code")
    )

    if not ok:
        raise RuntimeError("Continuation wash protocol failed.")

    print("Finished acid wash. Processing tissues...")


#%% Electrode helpers

def save_vm1_state(electrode):
    updates_path = os.path.join(inputFolder, "UPDATES.json")

    with open(updates_path, "r", encoding="utf-8") as f:
        conf = json.load(f)

    conf["electrode_state"]["vm1"] = electrode

    temp_path = updates_path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(conf, f, indent=2)

    os.replace(temp_path, updates_path)


def required_vm1_electrode(conf, test_code, lsv_side=None):
    vm1 = conf["electrode_config"]["vm1"]
    required_by_test = vm1["required_by_test"]

    if test_code == "L":
        if lsv_side is None:
            raise ValueError("LSV electrode request requires HER or OER side.")

        required = required_by_test["LSV"][lsv_side]

    else:
        test_name = TEST_CODE_TO_NAME[test_code]
        required = required_by_test[test_name]

    required = int(required)
    available = [int(x) for x in vm1["available"]]

    if required not in available:
        raise RuntimeError(
            f"VM1 electrode {required} was requested but is not listed "
            f"in available electrodes: {available}"
        )

    return required


def checked(code, action):
    if code != 0:
        raise RuntimeError(f"{action} failed; XArm code: {code}")


def move_vm1_to_cell(ws_url):
    """
    Lower the already-verified / already-seated VM1 electrode into the experiment
    well, then apply the same stabilization delay that used to live at the end
    of RoboticProcess.py.
    """
    code = """
Rpipette.move_to(target_well.bottom(180), speed=50)
Rpipette.move_to(target_well.bottom(plate_sel.rpipette_bottom_height))
protocol.delay(plate_sel.delay_seconds)
"""

    ok = asyncio.get_event_loop().run_until_complete(
        send_message_via_ws(ws_url, code, "vm1_to_measurement_cell")
    )

    if not ok:
        raise RuntimeError(
            "OT-2 failed to place VM1 electrode into experiment cell."
        )


def ensure_vm1_electrode(required, current, in_cell, ws_url):
    required = int(required)

    if current is not None:
        current = int(current)

    print(
        f"VM1 electrode check: current={current}, "
        f"required={required}, in_cell={in_cell}"
    )

    # Correct electrode already mounted. It enters solution only after this check.
    if current == required:
        print("Correct VM1 electrode already installed.")

        if not in_cell:
            move_vm1_to_cell(ws_url)

        return current, True

    if current is None:
        raise RuntimeError(
            "VM1 electrode state is None. "
            "Inspect the hardware before continuing."
        )

    if current not in OT2_ELECTRODES:
        raise RuntimeError(
            f"No OT-2 coordinates defined for electrode {current}."
        )

    if required not in OT2_ELECTRODES:
        raise RuntimeError(
            f"No OT-2 coordinates defined for electrode {required}."
        )

    if current not in XARM_ELECTRODES:
        raise RuntimeError(
            f"No XArm coordinates defined for current electrode {current}."
        )

    if required not in XARM_ELECTRODES:
        raise RuntimeError(
            f"No XArm coordinates defined for required electrode {required}."
        )


    # -------------------------------------------------------
    # OT-2 stages CURRENT electrode for XArm removal
    # -------------------------------------------------------

    old_ot2 = OT2_ELECTRODES[current]

    # If a test just finished, retract vertically before lateral movement.
    # If a wash just finished, the electrode is already raised at tcE z=110.
    if in_cell:
        safety_retract = """
Rpipette.move_to(target_well.bottom(180), speed=30)
"""
    else:
        safety_retract = ""

    stage_code = f"""
electrode_ref = res.reservoir_f['A1']

{safety_retract}

Rpipette.move_to(
    electrode_ref.bottom({old_ot2['hover_z']}).move(Point(x={old_ot2['x']}, y={old_ot2['y']})),
    speed=50
)

print("Electrode {current} staged for XArm removal.")
"""

    ok = asyncio.get_event_loop().run_until_complete(
        send_message_via_ws(ws_url, stage_code, f"stage_vm1_{current}")
    )

    if not ok:
        raise RuntimeError(
            f"OT-2 failed to stage electrode {current}."
        )


    # -------------------------------------------------------
    # XArm removes CURRENT electrode
    # -------------------------------------------------------

    arm = XArmAPI(XARM_IP)
    retreat_started = False
    old_arm = XARM_ELECTRODES[current]

    try:
        checked(arm.motion_enable(enable=True), "Enable XArm motion")
        checked(arm.set_mode(0), "Set XArm mode")
        checked(arm.set_state(0), "Set XArm state")

        # Explicit gripper initialization.
        checked(arm.set_gripper_enable(True), "Enable gripper")
        checked(arm.set_gripper_mode(0), "Set gripper position mode")
        checked(arm.set_gripper_speed(2000), "Set gripper speed")

        checked(
            arm.set_gripper_position(200, wait=True, auto_enable=True),
            "Open gripper for old electrode"
        )

        checked(
            arm.set_position(*old_arm["grab"], speed=50, mvacc=50, wait=True),
            f"Approach electrode {current}"
        )

        checked(
            arm.set_gripper_position(
                GRIP_POSITION,
                wait=True,
                timeout=5,
                auto_enable=True
            ),
            f"Grab electrode {current}"
        )

        checked(
            arm.set_position(*old_arm["seat"], speed=50, mvacc=50, wait=True),
            f"Store electrode {current}"
        )

        checked(
            arm.set_gripper_position(
                200,
                wait=True,
                timeout=5,
                auto_enable=True
            ),
            f"Release electrode {current}"
        )

        # At this exact point the P1000 physically has no electrode.
        save_vm1_state(None)


        # ---------------------------------------------------
        # XArm begins old-electrode retreat asynchronously
        # ---------------------------------------------------

        checked(
            arm.set_position(*old_arm["home"], speed=50, mvacc=50, wait=False),
            "Start old-electrode retreat"
        )

        retreat_started = True


        # ---------------------------------------------------
        # OT-2 picks REQUIRED electrode while XArm retreats
        # ---------------------------------------------------

        new_ot2 = OT2_ELECTRODES[required]

        pickup_code = f"""
electrode_ref = res.reservoir_f['A1']

Rpipette.move_to(
    electrode_ref.bottom({new_ot2['hover_z']}).move(Point(x={new_ot2['x']}, y={new_ot2['y']})),
    speed=50
)

Rpipette.move_to(
    electrode_ref.bottom({new_ot2['pick_z']}).move(Point(x={new_ot2['x']}, y={new_ot2['y']})),
    speed=20
)

Rpipette.move_to(
    electrode_ref.bottom({new_ot2['hover_z']}).move(Point(x={new_ot2['x']}, y={new_ot2['y']})),
    speed=20
)

print("Electrode {required} picked up and waiting for sturdy seating.")
"""

        ok = asyncio.get_event_loop().run_until_complete(
            send_message_via_ws(ws_url, pickup_code, f"pickup_vm1_{required}")
        )

        if not ok:
            raise RuntimeError(
                f"OT-2 failed to pick electrode {required}."
            )


        # ---------------------------------------------------
        # Finish old-electrode retreat before sturdy seating
        # ---------------------------------------------------

        checked(
            arm.set_position(*old_arm["home"], speed=50, mvacc=50, wait=True),
            "Finish old-electrode retreat"
        )

        retreat_started = False


        # ---------------------------------------------------
        # Sturdy installation of the NEW electrode
        # ---------------------------------------------------

        p = XARM_ELECTRODES[required]

        # p["grab"] = [x, y, z, roll, pitch, yaw]
        seat_pose = list(p["grab"])
        seat_pose[2] += SEAT_UP_MM
        seat_pose[4] += TIP_DOWN_PITCH_DEG
        seat_pose[5] += SEAT_YAW

        checked(
            arm.set_gripper_position(200, wait=True, auto_enable=True),
            "Open gripper for sturdy seating"
        )

        checked(
            arm.set_position(*p["home"], speed=80, mvacc=100, wait=True),
            "Move to new-electrode home"
        )

        checked(
            arm.set_position(*p["grab"], speed=150, mvacc=30, wait=True),
            "Move to new-electrode grab position"
        )

        checked(
            arm.set_gripper_position(
                GRIP_POSITION,
                wait=True,
                timeout=5,
                auto_enable=True
            ),
            "Close gripper"
        )

        code, actual_position = arm.get_gripper_position()
        checked(code, "Read gripper position")

        print(
            f"Gripper commanded to {GRIP_POSITION}; "
            f"reported position: {actual_position}"
        )

        checked(
            arm.set_position(*seat_pose, speed=20, mvacc=4, wait=True),
            "Seat and tilt electrode"
        )

        time.sleep(0.3)

        # Release before returning so the electrode is not pulled back down.
        checked(
            arm.set_gripper_position(
                200,
                wait=True,
                timeout=5,
                auto_enable=True
            ),
            "Release electrode after seating"
        )

        checked(
            arm.set_position(*p["grab"], speed=10, mvacc=15, wait=True),
            "Return after release"
        )

        checked(
            arm.set_position(*p["home"], speed=80, mvacc=100, wait=True),
            "Retreat home after seating"
        )

        print(
            f"Electrode {required}: seated upward by {SEAT_UP_MM} mm "
            f"with {TIP_DOWN_PITCH_DEG} degrees of pitch."
        )


        # ---------------------------------------------------
        # State becomes valid only after sturdy installation
        # ---------------------------------------------------

        save_vm1_state(required)

        print(
            f"VM1 electrode changed and seated: "
            f"{current} -> {required}"
        )


        # ---------------------------------------------------
        # ONLY NOW may the verified/seated electrode enter solution
        # ---------------------------------------------------

        move_vm1_to_cell(ws_url)

        return required, True


    finally:
        if retreat_started:
            try:
                arm.set_position(
                    *old_arm["home"],
                    speed=50,
                    mvacc=50,
                    wait=True
                )
            except Exception as e:
                print(
                    f"Warning: could not confirm XArm retreat: {e}"
                )

        try:
            arm.disconnect()
        except Exception:
            pass


#%% Individual test runners

def run_cv(outputFolder):
    try:
        stdout_text = cv_run(params=None, outdir=outputFolder)
        print(stdout_text)

    except CalledProcessError as e:
        print(e.stdout or "", file=sys.stderr)
        raise


def run_lsv_side(outputFolder, side):
    try:
        print(f"Starting {side} LSV...")

        stdout_text = lsv_initt.lsv_run(
            params=None,
            outdir=outputFolder,
            side=side
        )

        print(stdout_text)

        print(
            f"{side} LSV completed from NIMS."
        )

    except CalledProcessError as e:
        print(e.stdout or "", file=sys.stderr)
        raise


def run_eis(outputFolder, inputFolder, ws_url):
    avg_sigma, sigma_status = eis_run(
        outputFolder=outputFolder,
        inputFolder=inputFolder
    )
    if not math.isfinite(float(avg_sigma)):
        raise RuntimeError(f"EIS reps couldn't produce valid conductivity: {sigma_status}")
    print(f"EIS returned avg sigma = {avg_sigma}, status = {sigma_status}")
    return float(avg_sigma)


#%% Main sequence executor

def main():
    try:
        with open(inputFolder + "/UPDATES.json", "r") as conf_file:
            conf = json.load(conf_file)

    except Exception as e:
        print(f"couldnt locate UPDATES: {e}")
        raise

    tests = conf["tests"]
    sequence = normalize_tests(tests)

    psycho = conf["psycho"]

    electrode_config = conf.get("electrode_config")
    electrode_state = conf.get("electrode_state", {})

    if electrode_config is None:
        raise RuntimeError(
            "electrode_config is missing from UPDATES.json."
        )

    if "vm1" not in electrode_state:
        raise RuntimeError(
            "electrode_state.vm1 is missing from UPDATES.json."
        )

    current_vm1 = electrode_state["vm1"]

    # RoboticProcess now hands control to EXPRunner with the P1000
    # safely ABOVE the experiment well, not submerged.
    vm1_in_cell = False

    print(f"Electrochemical sequence selected: {sequence}")

    conductivity = None
    for test_code in sequence:

        print("")
        print("#" * 60)
        print(f"Starting test: {test_code}")
        print("#" * 60)


        # =======================================================
        # EIS
        # =======================================================

        if test_code == "E":

            required = required_vm1_electrode(
                conf,
                "E"
            )

            current_vm1, vm1_in_cell = ensure_vm1_electrode(
                required,
                current_vm1,
                vm1_in_cell,
                ws_url
            )

            conductivity = run_eis(outputFolder, inputFolder, ws_url)

            print("EIS completed from NIMS.")

            if WASH_AFTER_EIS:
                run_wash(ws_url)
                vm1_in_cell = False


        # =======================================================
        # CV
        # =======================================================

        elif test_code == "C":

            required = required_vm1_electrode(
                conf,
                "C"
            )

            current_vm1, vm1_in_cell = ensure_vm1_electrode(
                required,
                current_vm1,
                vm1_in_cell,
                ws_url
            )

            run_cv(outputFolder)

            print("CV completed from NIMS.")

            if WASH_AFTER_CV:
                run_wash(ws_url)
                vm1_in_cell = False


        # =======================================================
        # LSV
        # =======================================================

        elif test_code == "L":

            lsv_mode = str(
                lsv_initt.DEFAULT_PARAMS["lsv"]["lsv_mode"]
            ).upper()

            if lsv_mode not in {"HER", "OER", "BOTH"}:
                raise ValueError(
                    f"Invalid LSV mode: {lsv_mode}"
                )


            # ---------------------------------------------------
            # HER ONLY
            # ---------------------------------------------------

            if lsv_mode == "HER":

                required = required_vm1_electrode(
                    conf,
                    "L",
                    "HER"
                )

                current_vm1, vm1_in_cell = ensure_vm1_electrode(
                    required,
                    current_vm1,
                    vm1_in_cell,
                    ws_url
                )

                run_lsv_side(
                    outputFolder,
                    "HER"
                )

                if WASH_AFTER_LSV:
                    run_wash(ws_url)
                    vm1_in_cell = False

                lsv_initt.finalize_lsv_result(
                    outputFolder,
                    psycho,
                    "HER"
                )


            # ---------------------------------------------------
            # OER ONLY
            # ---------------------------------------------------

            elif lsv_mode == "OER":

                required = required_vm1_electrode(
                    conf,
                    "L",
                    "OER"
                )

                current_vm1, vm1_in_cell = ensure_vm1_electrode(
                    required,
                    current_vm1,
                    vm1_in_cell,
                    ws_url
                )

                run_lsv_side(
                    outputFolder,
                    "OER"
                )

                if WASH_AFTER_LSV:
                    run_wash(ws_url)
                    vm1_in_cell = False

                lsv_initt.finalize_lsv_result(
                    outputFolder,
                    psycho,
                    "OER"
                )


            # ---------------------------------------------------
            # BOTH: HER first, then OER
            # ---------------------------------------------------

            else:
                lsv_order = [str(side).upper()
                    for side in conf["electrode_config"]["vm1"]["lsv_order"]]

                if len(lsv_order) != 2 or set(lsv_order) != {"HER", "OER"}:
                    raise ValueError("lsv_order must contain HER and OER exactly once.")

                for side in lsv_order:
                    required = required_vm1_electrode(conf, "L", side)
                    current_vm1, vm1_in_cell = ensure_vm1_electrode(
                    required, current_vm1, vm1_in_cell, ws_url)

                    run_lsv_side(outputFolder, side)

                    if WASH_AFTER_LSV:
                        run_wash(ws_url)
                        vm1_in_cell = False

                lsv_initt.finalize_lsv_result(outputFolder, psycho, "BOTH", conductivity=conductivity)


            print(
                f"LSV {lsv_mode} completed from NIMS."
            )


        else:
            raise ValueError(
                f"Unexpected test code: {test_code}"
            )


        print("#" * 60)
        print(f"Finished test: {test_code}")
        print("#" * 60)


if __name__ == "__main__":
    main()
