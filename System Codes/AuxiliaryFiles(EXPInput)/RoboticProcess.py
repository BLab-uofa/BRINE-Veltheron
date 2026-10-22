import requests
import json
import asyncio
import websockets
import time
from websockets.exceptions import WebSocketException, ConnectionClosedError
import subprocess
import sys
import uuid



sys.path.append(inputFolder)


# ---------------- shared UPDATES.json liquid ledger helpers ==================================
def _blank(v):
    return v is None or v == "" or v == "null"

def _as_float_or_none(v):
    if _blank(v):
        return None
    return float(v)

def _as_list(value, n, default=None):
    if isinstance(value, list):
        out = value[:n]
    else:
        out = [value for _ in range(n)]
    while len(out) < n:
        out.append(default)
    return out

def _ensure_liquid_schema(conf):
    """Keep UPDATES.json backward compatible while adding the refill ledger."""
    n = int(conf.get("real_salt_no", len(conf.get("Salt_Tank", [])) or 1))

    salt_tank = _as_list(conf.get("Salt_Tank", 50000), n, 50000)
    water_tank = conf.get("Water_Tank", 50000)

    conf.setdefault("warning_margin_uL", 3000)
    conf.setdefault("protocol_refill_margin_uL", 1000)
    conf.setdefault("physical_tube_capacity_uL", 50000)

    req = conf.setdefault("refill_request", {})
    req.setdefault("enabled", False)
    req["salt_uL"] = _as_list(req.get("salt_uL", [None] * n), n, None)
    req.setdefault("water_uL", None)

    state = conf.setdefault("liquid_state", {})
    state["salt_fill_uL"] = [float(x) for x in _as_list(state.get("salt_fill_uL", salt_tank), n, 50000)]
    state["salt_refill_at_usage_uL"] = [float(x) for x in _as_list(state.get("salt_refill_at_usage_uL", [0] * n), n, 0)]
    state["water_fill_uL"] = float(state.get("water_fill_uL", water_tank))
    state["water_refill_at_usage_uL"] = float(state.get("water_refill_at_usage_uL", 0))
    return conf

def _current_usage_from_updates(conf):
    n = int(conf.get("real_salt_no", len(conf.get("input_column_sums", [])) or 1))
    salt_used = [float(x) for x in _as_list(conf.get("input_column_sums", [0] * n), n, 0)]
    cell_vol = float(conf.get("cell_volume_uL", 0))
    cycle_no = int(conf.get("psycho", 0))
    water_used = max(0.0, cycle_no * cell_vol - sum(salt_used))
    return salt_used, water_used

def _apply_refill_request_if_enabled(conf):
    """
    If the user edited UPDATES.json and set refill_request.enabled = true,
    commit the refill into liquid_state and turn the trigger off.
    """
    conf = _ensure_liquid_schema(conf)
    req = conf["refill_request"]
    if not bool(req.get("enabled", False)):
        return conf, False

    salt_used, water_used = _current_usage_from_updates(conf)
    state = conf["liquid_state"]
    changed = False

    for i, v in enumerate(req.get("salt_uL", [])):
        refill_uL = _as_float_or_none(v)
        if refill_uL is not None:
            state["salt_fill_uL"][i] = refill_uL
            state["salt_refill_at_usage_uL"][i] = salt_used[i]
            changed = True

    water_refill_uL = _as_float_or_none(req.get("water_uL"))
    if water_refill_uL is not None:
        state["water_fill_uL"] = water_refill_uL
        state["water_refill_at_usage_uL"] = water_used
        changed = True

    req["enabled"] = False
    return conf, changed
# ---------------------------------------------------------------------------


def main():
    
    
    try:
        with open(inputFolder + '/UPDATES.json', 'r') as conf_file:
            conf = json.load(conf_file)
    except Exception as e:
        print('couldnt locate UPDATES')
        
    
    psycho = conf['psycho']
    suggest_ratios = conf['suggest_ratios']
    real_salt_no = conf['real_salt_no']
    input_column_sums = conf['input_column_sums']
    tests = conf['tests']
    Salt_Tank = conf['Salt_Tank']          # original/default values; kept for compatibility
    Water_Tank = conf['Water_Tank']        # original/default value; kept for compatibility
    Liquid_State = conf['liquid_state']    # effective fill + refill baseline used by OT2 height logic
    Physical_Tube_Capacity_UL = conf.get('physical_tube_capacity_uL', 50000)
    cell_max = conf['cell_volume_uL']
    stock_Ms = conf['stock_molarities']
    
    
    notebook_url = 'http://169.254.177.16:48888/api/sessions'
    headers = {'Content-Type': 'application/json'}
    session_data = {
        "kernel": {"name": "python3"},
        "name": "BRINE_TH",
        "path": "BRINE_TH.ipynb",
        "type": "notebook"
    }

    response = requests.post(notebook_url, headers=headers, data=json.dumps(session_data))
    session_info = response.json()
    if 'kernel' not in session_info:
        print("Error: Kernel information not found in the response.")
        exit()
        
    kernel_id = session_info['kernel']['id']
    ws_url = f"ws://169.254.177.16:48888/api/kernels/{kernel_id}/channels"

    

    def shutdown_kernel(kernel_id):
        shutdown_url = f"http://169.254.177.16:48888/api/kernels/{kernel_id}"
        requests.delete(shutdown_url)
        print("Kernel has been shut down.")

    async def send_message_via_ws(ws_url, code, message_name):
        try:
            print(f"Starting WebSocket execution for {message_name}...")
            async with websockets.connect(ws_url, open_timeout=60) as websocket:
                msg_id = str(uuid.uuid4())
                
                msg = {
                    "header": {
                        "msg_id": msg_id,
                        "username": "Reza",
                        "session": str(uuid.uuid4()),
                        "msg_type": "execute_request",
                        "version": "5.3"},
                    
                    "parent_header": {},
                    "metadata": {},
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": True,
                        "user_expressions": {},
                        "allow_stdin": False
                    },
                    "channel": "shell"
                }
                # Send the message
                await websocket.send(json.dumps(msg))
                print(f"Message sent for {message_name}. Waiting for response...")
                # Listen for responses until execution is fully completed
                while True:
                    raw = await websocket.recv()
                    response = json.loads(raw)
                    parent = response.get("parent_header", {})
                    if parent.get("msg_id") != msg_id:
                        continue
                    
                    print(f"{message_name} response: {raw}")
                    
                    if response.get("msg_type") == "error":
                        raise RuntimeError(response.get("content", {}))

                    if response.get("msg_type") == "execute_reply":
                        print(f"{message_name} execution completed.")
                        return True
                    
                    
                    # Check for the "idle" state indicating completion
                    if '"execution_state": "idle"' in response:
                        print(f"{message_name} execution completed, closing WebSocket.")
                        break
                await websocket.close()
                print(f"WebSocket closed for {message_name}.")
                return True
                
        except Exception as e:
            import traceback
            print(f"WebSocket/general error during {message_name}: {repr(e)}")
            traceback.print_exc()
            return False
    
    RESERVOIR_MODE = "50ml"   # "50ml" or "15ml"
    PLATE_MODE = "48well"     # "48well" or "96well"

    sol_prep = f"""   
    cycle_no = {psycho}
    f_VOL_string = {suggest_ratios}
    dimension = {real_salt_no}

    # These are mainly needed for the 50 mL conical reservoir height logic.
    # For 15 mL reservoir mode, they're NOTT used.
    col_sums = {input_column_sums}
    Salt_Tank = {Salt_Tank}
    Water_Tank = {Water_Tank}
    Liquid_State = {Liquid_State}
    Physical_Tube_Capacity_UL = {Physical_Tube_Capacity_UL}

    reservoir_mode = "{RESERVOIR_MODE}"
    plate_mode = "{PLATE_MODE}"

    TIPS_PER_RACK = 96 
    ROWS_PER_COL_RACK = 8
    PIP_MAX = 300
    PROTOCOL_REFILL_MARGIN = 1000

    import opentrons.execute
    protocol = opentrons.execute.get_protocol_api('2.20')
    from opentrons.types import Point, Location
    import json

    protocol.home()

    # Load pipettes
    Lpipette = protocol.load_instrument('p300_single_gen2', 'left')
    Rpipette = protocol.load_instrument('p1000_single_gen2', 'right')


    class ReservoirSelector:
        def __init__(self, protocol, mode, col_sums=None, Salt_Tank=None, Water_Tank=None,
                     Liquid_State=None, Physical_Tube_Capacity_UL=50000):
            self.protocol = protocol
            self.mode = mode
            self.col_sums = col_sums
            self.Salt_Tank = Salt_Tank
            self.Water_Tank = Water_Tank
            self.Liquid_State = Liquid_State or {{}}
            self.Physical_Tube_Capacity_UL = float(Physical_Tube_Capacity_UL)
            self.PROTOCOL_REFILL_MARGIN = 1000

            if mode == "15ml":
                self.reservoir = protocol.load_labware("nest_12_reservoir_15ml", 4)
                self.reservoir_c = protocol.load_labware("nest_12_reservoir_15ml", 11)
                # self.reservoir_c.set_offset(x=2.9, y=0, z=0) #USED T O BE +0.1
                self.reservoir_f = protocol.load_labware("nest_12_reservoir_15ml", 10)
                
                # For making it every-other-well
                self._RES15_WELLS = ["A1", "A3", "A5", "A7", "A9"]

            elif mode == "50ml":
                self.Res6_1 = protocol.load_labware("opentrons_6_tuberack_nest_50ml_conical", 3)
                self.Res6_1.set_offset(x=-8.5, y=0, z=0)

                self.Res6_2 = protocol.load_labware("opentrons_6_tuberack_nest_50ml_conical", 4)
                self.Res6_2.set_offset(x=-8.5, y=0, z=0)

                self.reservoir_c = protocol.load_labware("nest_12_reservoir_15ml", 11)
                self.reservoir_c.set_offset(x=1.2, y=0, z=0) #USED T O BE +0.1

                self.reservoir_f = protocol.load_labware("nest_12_reservoir_15ml", 10)

                self._RES1 = ["A1", "A2", "A3", "B1", "B2", "B3"]
                self._RES2 = ["A1", "A2", "A3", "B1", "B2"]  # B3 reserved for water

            else:
                raise ValueError("reservoir_mode must be either '15ml' or '50ml'.")

        def salt_source(self, i):
            if self.mode == "15ml":
                return self.reservoir[self._RES15_WELLS[i]]         # used to be self.reservoir["A" + str(i + 1)]

            idx = i + 1
            if idx <= len(self._RES1):
                return self.Res6_1[self._RES1[idx - 1]]

            idx2 = idx - len(self._RES1)

            if idx2 <= len(self._RES2):
                return self.Res6_2[self._RES2[idx2 - 1]]

            raise ValueError("Too many salt inputs: only 11 non-water 50 mL tube positions supported.")

        def water_source(self):
            if self.mode == "15ml":
                return self.reservoir["A12"]

            return self.Res6_1["B3"]  ### MAKE IT Res6_2

        def waste_source(self):
            if self.mode == "15ml":
                return self.reservoir["A11"]

            return self.reservoir_c["A6"]

        def needs_height_control(self):
            return self.mode == "50ml"

        def _bucket_height_from_remainder(self, r):
            if r < 10000:
                return 80
            elif r <= 20000:
                return 60
            elif r <= 25000:
                return 56
            elif r <= 30000:
                return 45
            elif r <= 40000:
                return 25
            else:
                return 4

        def _used_since_refill_salt(self, i):
            baselines = self.Liquid_State.get("salt_refill_at_usage_uL", [0] * len(self.col_sums))
            return max(0.0, float(self.col_sums[i]) - float(baselines[i]))

        def _used_since_refill_water(self, cycle_no, target_total_volume):
            total_salts = sum(self.col_sums)
            water_used = max(0.0, cycle_no * float(target_total_volume) - total_salts)
            baseline = float(self.Liquid_State.get("water_refill_at_usage_uL", 0))
            return max(0.0, water_used - baseline)

        def _height_from_fill_and_used(self, fill_uL, used_since_refill_uL):
            # Convert a partial refill into the equivalent "consumed from a full 50 mL tube".
            # Example: if you refill only 20 mL, the starting meniscus is already lower than 50 mL.
            remaining = max(0.0, float(fill_uL) - float(used_since_refill_uL))
            equivalent_consumed_from_full = max(
                0.0,
                self.Physical_Tube_Capacity_UL - remaining
            )
            return self._bucket_height_from_remainder(equivalent_consumed_from_full)

        def salt_height(self, i):
            if self.mode == "15ml":
                return None

            fill_uL = self.Liquid_State.get("salt_fill_uL", self.Salt_Tank)[i]
            used_since = self._used_since_refill_salt(i)
            return self._height_from_fill_and_used(fill_uL, used_since)

        def water_height(self, cycle_no, target_total_volume):
            if self.mode == "15ml":
                return None

            fill_uL = self.Liquid_State.get("water_fill_uL", self.Water_Tank)
            used_since = self._used_since_refill_water(cycle_no, target_total_volume)
            return self._height_from_fill_and_used(fill_uL, used_since)

        def aspirate_salt(self, pipette, amount, i):
            src = self.salt_source(i)

            if self.mode == "50ml":
                h = self.salt_height(i)
                pipette.aspirate(amount, src.bottom(h))
            else:
                pipette.aspirate(amount, src)

        def aspirate_water(self, pipette, amount, cycle_no, target_total_volume):
            src = self.water_source()

            if self.mode == "50ml":
                h = self.water_height(cycle_no, target_total_volume)
                pipette.aspirate(amount, src.bottom(h))
            else:
                pipette.aspirate(amount, src)


    class PlateSelector:
        def __init__(self, protocol, mode):
            self.protocol = protocol
            self.mode = mode

            if mode == "48well":
                self.WELLS_PER_PLATE = 48
                self.ROWS_PER_COL_PLATE = 6
                self.target_total_volume = {cell_max}
                self.rpipette_bottom_height = 76
                self.mix_volume = 200
                self.delay_seconds = 120
                self.chunk_large_volumes = True
                self.plate_order = ["8", "5", "8", "5"]

            elif mode == "96well":
                self.WELLS_PER_PLATE = 96
                self.ROWS_PER_COL_PLATE = 8
                self.target_total_volume = {cell_max}
                self.rpipette_bottom_height = 68
                self.mix_volume = 200
                self.delay_seconds = 30
                self.chunk_large_volumes = False
                self.plate_order = ["5", "8"]

            else:
                raise ValueError("plate_mode must be either '48well' or '96well'.")

        def get_plate(self, cycle_no):
            plate_index = (cycle_no - 1) // self.WELLS_PER_PLATE

            if plate_index >= len(self.plate_order):
                raise ValueError(
                    "Test cells exceed the available plates. "
                    "Add more plates or cut down the experiment count."
                )

            deck_slot = self.plate_order[plate_index]

            if self.mode == "48well":
                with open("blabvertex_48_wellplate_1400ul.json") as f:
                    labdef = json.load(f)

                if not self.protocol.deck.get(deck_slot):
                    plate = self.protocol.load_labware_from_definition(labdef, deck_slot)
                else:
                    plate = self.protocol.deck[deck_slot]
                    
                plate.set_offset(x=0.7, y=-1.5, z=0.0)

            else:
                if not self.protocol.deck.get(deck_slot):
                    plate = self.protocol.load_labware("nest_96_wellplate_2ml_deep", deck_slot)
                else:
                    plate = self.protocol.deck[deck_slot]

                plate.set_offset(x=-0.5, y=0.0, z=0.0)

            return plate

        def target_well(self, cycle_no):
            plate = self.get_plate(cycle_no)
            rows = plate.rows()

            pos_in_plate = (cycle_no - 1) % self.WELLS_PER_PLATE
            row_no = pos_in_plate % self.ROWS_PER_COL_PLATE
            col_no = pos_in_plate // self.ROWS_PER_COL_PLATE

            return rows[row_no][col_no]


    def get_tip_rack_and_address(tip_no):
        tip_rack_order = ['9', '6', '2', '1', '4', '9', '6']

        rack_index = tip_no // TIPS_PER_RACK

        if rack_index >= len(tip_rack_order):
            raise ValueError("Tip number exceeds available tip racks.")

        deck_slot = tip_rack_order[rack_index]

        if not protocol.deck.get(deck_slot):
            Tips = protocol.load_labware('opentrons_96_tiprack_300ul', deck_slot)
            Tips.set_offset(x=-0.5, y=1, z=0)
        else:
            Tips = protocol.deck[deck_slot]

        position_in_rack = tip_no % TIPS_PER_RACK
        row = position_in_rack % ROWS_PER_COL_RACK
        col = position_in_rack // ROWS_PER_COL_RACK
        tip_address = chr(65 + row) + str(col + 1)

        return Tips, tip_address


    def de_cling_tip(pip):
        pip.touch_tip(v_offset=-5, speed=50)
        pip.touch_tip(v_offset=-5, speed=50)
        pip.touch_tip(radius=1.05, v_offset=-2, speed=80)


    # Initialize selectors
    res = ReservoirSelector(
        protocol,
        reservoir_mode,
        col_sums=col_sums,
        Salt_Tank=Salt_Tank,
        Water_Tank=Water_Tank,
        Liquid_State=Liquid_State,
        Physical_Tube_Capacity_UL=Physical_Tube_Capacity_UL
    )

    plate_sel = PlateSelector(protocol, plate_mode)

    # Main Protocol STARTS HERE
    target_well = plate_sel.target_well(cycle_no)

    vol = [int(round(float(x))) for x in f_VOL_string]
    w_vol = plate_sel.target_total_volume - sum(vol)

    if w_vol < -1:
        raise ValueError("Total salt volume is larger than target well volume.")

    # Tip number logic from 50 mL protocol.
    # If dimension == 1, only one salt tip is needed.
    tip_no = (cycle_no - 1) * (dimension + 1) if dimension > 1 else (cycle_no - 1) * dimension

    t = 0


    def dispense_steps(steps, tip_offset_start=0):
        global t

        for amount, src, dest, mix_after, inc, src_kind, src_index, use_existing_tip, keep_tip in steps:

            if amount < 10:
                continue
            elif amount < 20:
                amount = 20

            if not use_existing_tip:
                rack, idx = get_tip_rack_and_address(tip_no + tip_offset_start + t)
                Lpipette.pick_up_tip(rack[idx])

            if plate_sel.chunk_large_volumes:
                chunks = []
                remaining = amount

                while remaining > 0:
                    chunk = min(remaining, PIP_MAX)
                    chunks.append(chunk)
                    remaining -= chunk
            else:
                chunks = [amount]

            for n, chunk in enumerate(chunks):
                last_chunk = n == len(chunks) - 1

                if src_kind == "salt":
                    res.aspirate_salt(Lpipette, chunk, src_index)

                    if res.needs_height_control():
                        de_cling_tip(Lpipette)

                elif src_kind == "water":
                    res.aspirate_water(
                        Lpipette,
                        chunk,
                        cycle_no,
                        plate_sel.target_total_volume
                    )

                else:
                    raise ValueError("Unknown source kind in dispense_steps.")

                if res.needs_height_control():
                    Lpipette.move_to(src.bottom(130))
                else:
                    Lpipette.move_to(src.top())

                if plate_sel.chunk_large_volumes and not last_chunk:
                    Lpipette.dispense(chunk, dest.top())
                else:
                    Lpipette.dispense(chunk, dest)
                    Lpipette.blow_out(location=dest.top())

            if mix_after:
                reps, mix_vol = mix_after
                Lpipette.mix(reps, mix_vol, location=dest)

            Lpipette.move_to(dest.top())

            if not keep_tip:
                Lpipette.drop_tip()

            if inc:
                t += 1


    # Mixing begins here
    if w_vol >= 20:

        # Water first, keep same tip for first salt if available.
        ops = [
            (
                w_vol,
                res.water_source(),
                target_well,
                None,
                False,
                "water",
                None,
                False,
                True
            )
        ]

        salt_steps = [
            (
                vol[i],
                res.salt_source(i),
                target_well,
                (4, plate_sel.mix_volume) if i == dimension - 1 else None,
                True,
                "salt",
                i,
                False,
                False
            )
            for i in range(dimension)
            if vol[i] >= 20
        ]

        if salt_steps:
            # First salt reuses the water tip.
            salt_steps[0] = (
                salt_steps[0][0],
                salt_steps[0][1],
                salt_steps[0][2],
                salt_steps[0][3],
                salt_steps[0][4],
                salt_steps[0][5],
                salt_steps[0][6],
                True,
                False
            )
        else:
            # No salt step exists, so water tip must be dropped.
            ops[0] = (
                ops[0][0],
                ops[0][1],
                ops[0][2],
                ops[0][3],
                ops[0][4],
                ops[0][5],
                ops[0][6],
                ops[0][7],
                False
            )

        ops += salt_steps
        dispense_steps(ops)


    elif 0 < w_vol < 20:

        rack, idx = get_tip_rack_and_address(tip_no)
        Lpipette.pick_up_tip(rack[idx])

        # Trick to work with water volume less than 20 uL.
        # Use 150 uL for 96-well plate, 200 uL for 48-well plate.
        helper_vol = plate_sel.mix_volume

        if reservoir_mode == "50ml":
            h_w = res.water_height(cycle_no, plate_sel.target_total_volume)
            Lpipette.move_to(res.water_source().bottom(h_w))
            Lpipette.aspirate(helper_vol)
            Lpipette.move_to(res.water_source().bottom(130))
        else:
            Lpipette.aspirate(helper_vol, res.water_source())
            Lpipette.move_to(res.water_source().top())

        Lpipette.dispense(helper_vol, target_well)

        Lpipette.aspirate(helper_vol - w_vol, target_well)
        Lpipette.move_to(target_well.top())
        Lpipette.dispense(helper_vol - w_vol, res.waste_source())
        Lpipette.move_to(res.waste_source().top())

        # Keep the same tip for the first salt.
        t = 0

        first_salt_idx = next((i for i in range(dimension) if vol[i] >= 20),None)

        ops = [(
                vol[i],
                res.salt_source(i),
                target_well,
                (4, plate_sel.mix_volume) if i == dimension - 1 else None,
                True,
                "salt",
                i,
                True if i == first_salt_idx else False,
                False
            )
            for i in range(dimension)
            if vol[i] >= 20
        ]

        if not ops:
            Lpipette.drop_tip()


        dispense_steps(ops, tip_offset_start=0)


    else:
        # No water needed.
        ops = []

        for i in range(dimension):
            mix = (4, plate_sel.mix_volume) if i == dimension - 1 and dimension != 1 else None
            inc = i != dimension - 1

            ops.append(
                (
                    vol[i],
                    res.salt_source(i),
                    target_well,
                    mix,
                    inc,
                    "salt",
                    i,
                    False,
                    False
                )
            )

        dispense_steps(ops)


    Rpipette.move_to(target_well.bottom(180)) #HIEGHT WAS plate_sel.rpipette_bottom_height
    # protocol.delay(plate_sel.delay_seconds)
    """     
    
    
    # SEND PROTOCOL
    print("Start WebSocket: Solution prep ongoing...")
    
    ok = asyncio.get_event_loop().run_until_complete(send_message_via_ws(ws_url, sol_prep, 'sol_prep'))
    if not ok:
        raise RuntimeError("Solution prep failed. WebSocket never connected/executed.")

    print("Solution prep done...")
    time.sleep(0.5)
    
    
    
    #%% Electrochemical Test and Wash
    try:
        exec_globals = {
            "__name__": "__main__",
            "outputFolder": outputFolder,
            "inputFolder": inputFolder,
            "ws_url": ws_url
        }

        with open(inputFolder + "/EXPRunner_sequence.py", "r") as file:
            code = file.read()

        exec(code, exec_globals)

    except Exception as e:
        print(f"Error during electrochemical sequence: {e}")
        sys.exit(0)
    
    
    #%% CLEAN UP
    shutdown_kernel(kernel_id)

if __name__ == "__main__":
    main()
