from __future__ import annotations
import numpy as np
import nimsos.ai_tools
import nimsos.input_tools
import nimsos.output_tools
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, List, Callable, Dict
import pandas as pd
import matplotlib.pyplot as plt

class selection():
    """Class of AI selection

    This class can select the next candidates depending on the AI methods.

    """

    def __init__(self, method, input_file, output_file, output_folder, num_objectives, num_proposals, num_inputs, Psycho):
        """Constructor
        
        This function do not depend on robot.

        Args:
            method (str): "RE" or "BO"or "BLOX" or "PDC"
            input_file (str): the file for candidates for AI algorithm
            output_file (str): the file for proposals from AI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
            num_inputs (int): the number of fking inputs

        """

        self.method = method
        self.input_file = input_file
        self.output_file = output_file
        self.output_folder = output_folder
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals
        self.num_inputs = num_inputs
        self.Psycho = Psycho
        
        res = self.module_selection()
        


    def module_selection(self):
        """module selection of preparation input
        
        This function do not depend on robot.

        Returns:
            res (str): True for success, False otherwise.

        """
        res = 'False'
        
        if self.method == "Ax": #note Ax also needs Output folder bc it saves pareto plots in addition to just the exp conditions
            res = nimsos.ai_tools.ai_tool_Ax.AxV112(self.input_file, self.output_file, self.output_folder,
            self.num_objectives, self.num_proposals, self.num_inputs, self.Psycho).select()
            return res

        if self.method == "predefined":
            res = nimsos.ai_tools.ai_tool_predefined.Predefined(self.input_file,self.output_file,self.output_folder,
            self.num_objectives, self.num_proposals, self.num_inputs, self.Psycho, force_start=True).select()
            return res
        
        if self.method == "RE":
            res = nimsos.ai_tools.ai_tool_re.RE(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals).select()
            return res

        if self.method == "PHYSBO":
            res = nimsos.ai_tools.ai_tool_physbo.PHYSBO(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals).select()
            return res

        if self.method == "PDC":
            res = nimsos.ai_tools.ai_tool_pdc.PDC(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals).select()
            return res

        if self.method == "BLOX":
            res = nimsos.ai_tools.ai_tool_blox.BLOX(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals).select()
            return res
        
        if self.method == "GAUCHE":
            res = nimsos.ai_tools.ai_tool_gauche.GAUCHE(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals).select()
            return res
        
        if self.method == "SMAC3":
            res = nimsos.ai_tools.ai_tool_smac3.SMAC3(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.num_inputs).select()
            return res
        
        if self.method == "SMAC3_EXPLORE":
            res = nimsos.ai_tools.ai_tool_smac3_explorer.SMAC3(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.num_inputs).select()
            return res
        
        if self.method == "SMAC3_EXPLOIT_REGION":
            res = nimsos.ai_tools.ai_tool_smac3_exploiter_regionbased.SMAC3(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.num_inputs).select()
            return res

        if self.method == "INDIVIDUALS":
            res = nimsos.ai_tools.ai_tool_individuals.INDIVIDUALS(self.input_file, self.output_file, self.output_folder,
            self.num_objectives, self.num_proposals, self.num_inputs).select()
            return res
            


class preparation_input():
    """Class of preparation input

    This class can create input for robot experiments and star robot experiments.

    """

    def __init__(self, machine, input_file, input_folder, output_folder, Real_Inputs, EC_Tests, Salt_Tank, Water_Tank, cell_volume_uL,
                 stock_molarities, Psycho, Electrode_Config=None):
        """Constructor
        
        This function do not depend on robot.

        Args:
            machine (str): "STAN" or "NAREE"
            input_file (str): the file for proposals from MI algorithm
            inputFolder (str): the folder where input files for robot are stored

        """

        self.machine = machine
        self.input_file = input_file
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.Real_Inputs = Real_Inputs
        self.EC_Tests = EC_Tests
        self.Salt_Tank = Salt_Tank
        self.Water_Tank = Water_Tank
        self.Psycho = Psycho
        self.cell_volume_uL = cell_volume_uL
        self.stock_molarities = stock_molarities
        self.Electrode_Config = Electrode_Config
        
        res = self.module_selection()



    def module_selection(self):
        """module selection of preparation input
        
        This function do not depend on robot.

        Returns:
            res (str): True for success, False otherwise.
        """

        if self.machine == "STAN":
            res = nimsos.input_tools.preparation_input_standard.Standard(self.input_file, self.input_folder, self. output_folder, self.Real_Inputs,
                                                                         self.EC_Tests, self.Salt_Tank, self.Water_Tank,
                                                                         self.cell_volume_uL, self.stock_molarities, self.Electrode_Config).perform(self.Psycho)
            return res

        if self.machine == "NAREE":
            res = nimsos.input_tools.preparation_input_naree.NAREE(self.input_file, self.input_folder).perform()
            return res



class analysis_output():
    """Class of analysis output

    This class can analyze output.

    """

    def __init__(self, machine, input_file, num_objectives, num_proposals, output_folder, Psycho, objectives_info = None):
        """Constructor
        
        This function do not depend on robot.

        Args:
            machine (str): "STAN" or "NAREE"
            input_file (str): the file for proposals from MI algorithm
            output_file (str): the file for candidates which will be updated in this script
            num_objectives (int): the number of objectives
            output_folder (str): the folder where the output files are stored by robot
            objectives_select (dict): the dictionary for objectives selection

        """

        self.machine = machine
        self.input_file = input_file
        self.num_objectives = num_objectives
        self.num_proposals= num_proposals
        self.output_folder = output_folder
        self.objectives_info = objectives_info
        self.Psycho = Psycho

        res = self.module_selection()


    def module_selection(self):
        """module selection of analysis input
        
        This function do not depend on robot.

        Returns:
            res (str): True for success, False otherwise.


        """

        if self.machine == "STAN":
            res = nimsos.output_tools.analysis_output_standard.Standard(self.input_file, self.num_objectives, self.num_proposals, self.output_folder).perform(self.Psycho)
            return res
        if self.machine == "NAREE":
            res = nimsos.output_tools.analysis_output_naree.NAREE(self.input_file, self.output_file, self.num_objectives, self.output_folder, self.objectives_info).perform()
            return res

class digital_twin():
    """
    NIMSOS-facing Digital Twin entry point.

    Placement in main loop:
        selection()
        feedback()
        digital_twin()
        preparation_input()
        analysis_output()
    """

    def __init__(
        self,
        input_file,
        output_folder,
        num_inputs,
        num_objectives,
        num_proposals,
        Psycho,
        cell_volume_uL,
        stock_molarities,
        salt_formulas,
        phreeqc_exe_path,
        phreeqc_database_path,
        mobility_file,
        objective_map,
        temp_C=25.0,
        Real_Inputs=None,
        Salt_Tank=None,
        Water_Tank=None,
        EC_Tests=None):
        
        self.input_file = input_file
        self.output_folder = output_folder
        self.num_inputs = int(num_inputs)
        self.num_objectives = int(num_objectives)
        self.num_proposals = int(num_proposals)
        self.Psycho = int(Psycho)
        
        self.cell_volume_uL = cell_volume_uL
        self.stock_molarities = stock_molarities
        self.salt_formulas = salt_formulas
        
        self.phreeqc_exe_path = phreeqc_exe_path
        self.phreeqc_database_path = phreeqc_database_path
        self.objective_map = objective_map
        self.temp_C = temp_C 

        self.mobility_file = mobility_file

        self.Real_Inputs = Real_Inputs
        self.Salt_Tank = Salt_Tank
        self.Water_Tank = Water_Tank
        self.EC_Tests = EC_Tests


        
        res = self.module_selection()

    def module_selection(self):
        from nimsos.digital_twin.runner import DigitalTwinRunner
        from nimsos.digital_twin.config import DTConfig

        cfg = DTConfig(
            input_file=self.input_file,
            output_folder=self.output_folder,
            num_inputs=self.num_inputs,
            num_objectives=self.num_objectives,
            num_proposals=self.num_proposals,
            Psycho=self.Psycho,
            Real_Inputs=self.Real_Inputs,
            Salt_Tank=self.Salt_Tank,
            Water_Tank=self.Water_Tank,
            EC_Tests=self.EC_Tests,
            cell_volume_uL=self.cell_volume_uL,
            stock_molarities=self.stock_molarities,
            salt_formulas=self.salt_formulas,
            phreeqc_exe_path=self.phreeqc_exe_path,
            phreeqc_database_path=self.phreeqc_database_path,
            
            objective_map= self.objective_map,
            temp_C = self.temp_C,
            mobility_file=self.mobility_file)

        runner = DigitalTwinRunner(cfg)
        return runner.run()


class feedback():
    """
    Two-way Telegram feedback guardrail.  (Also monitors a virtual "Water" input computed as: Psycho*1100 - sum(all input columns).)
    - Run after selection, before preparation_input.
    - Per-input thresholds based on tank capacity (Salt_Tank / Water_Tank), increasing by one full tank per dismiss/continue.    - If over threshold: warn and offer Stop / Pause / Dismiss.
    - Pause opens a no-timeout menu for Continue / Stop.
    - No-response streak across cycles: after 3, auto-open Pause menu.
    - 1-minute catch window after sending the warning to capture immediate taps.
    - Logs all Telegram updates to nimsos_feedback_updates.log.
    - Sends a quick ack ("received: ...") when a command is seen.
    """

    def __init__(self, input_file, input_folder, num_inputs, num_proposals, Psycho, Salt_Tank, Water_Tank, tg_bot_token=None, tg_chat_id=None):
        import os, json, time, requests, csv

        self.input_file = input_file
        self.input_folder = input_folder
        self.num_inputs = int(num_inputs)
        self.num_proposals = int(num_proposals)
        self.Psycho = int(Psycho)
        self.Salt_Tank = Salt_Tank
        self.Water_Tank = Water_Tank

        # Telegram creds
        if not tg_bot_token or not tg_chat_id:
            print("[feedback] Telegram token/chat_id not provided; skipping feedback.")
            return
        self.token = tg_bot_token
        self.chat_id = int(tg_chat_id)
        self.base = f"https://api.telegram.org/bot{self.token}"

        # Files
        self.root_dir = os.path.dirname(os.path.abspath(self.input_file))
        self.input_folder = os.path.normpath(self.input_folder) # added this as UPDATES is in EXPInput
        self.updates_path = os.path.join(self.root_dir, self.input_folder, "UPDATES.json")
        self.state_path = os.path.join(self.root_dir, "nimsos_feedback_state.json")
        self.log_path = os.path.join(self.root_dir, "nimsos_feedback_updates.log")

        # Load / init state
        self.state = self._load_state()

        # UPDATES.json is now the single source of truth for liquid amount.
        # If the user manually enabled a refill request between cycles,
        # commit it here so feedback and OT2 share the same reset point.
        try:
            conf = self._load_updates_conf()
            conf, changed = self._apply_refill_request_if_enabled(conf)
            if changed:
                self._save_updates_conf(conf)
        except Exception as e:
            print(f"[feedback] UPDATES liquid ledger unavailable: {e}")

        # 1) One-off initiation message at Psycho == 1
        self._send_text(f"🧪 Experiment initiated (cycle {self.Psycho}).")
        if self.Psycho == 1 and not self.state.get("initiated_sent"):
            self.state["initiated_sent"] = True
            self._save_state()

        # 2) Ingest any pending commands at the very start of this cycle
        received_cmd = self._poll_and_select_best()  # may be None
        if received_cmd:
            self._ack_cmd(received_cmd)              # visible immediate bot ack
            if self._apply_command_at_cycle_start(received_cmd):
                return  # handled (e.g., paused or stopped); nothing else to do in this call

        # 3) Compute per-input cumulative usage and compare with thresholds
        try:
            usage, input_names = self._compute_usage(csv)
        except Exception as e:
            print(f"[feedback] Unable to read candidates file: {e}")
            return

        flagged = self._find_flagged_inputs(usage)

        if not flagged:
            # Nothing to do; clear any previous pending action if usage dropped below thresholds
            self.state["awaiting_action"] = False
            self.state["pending_inputs"] = []
            self.state["no_action_streak"] = 0
            self._save_state()
            return

        # Maintain pending set & awaiting flag
        if not self.state.get("awaiting_action"):
            self.state["awaiting_action"] = True
            self.state["pending_inputs"] = flagged
            self.state["no_action_streak"] = 0
            self._save_state()
        else:
            # refresh pending to current flags (e.g., new inputs also crossed)
            self.state["pending_inputs"] = flagged
            self._save_state()

        # 4) Send warning with buttons
        low_labels = ", ".join(input_names[i] for i in self.state["pending_inputs"])
        title = "⚠️ Resource low"
        body = f"Inputs over threshold: {low_labels}\nCycle: {self.Psycho}\n" \
               f"Options: Stop / Pause / Dismiss (Dismiss/Continue raises threshold for those inputs)."
        self._send_warning_with_buttons(title, body)

        # 4a) 1-minute catch window: poll repeatedly to capture immediate taps
        caught_cmd = self._catch_window(seconds=60)
        if caught_cmd:
            self._ack_cmd(caught_cmd)
            handled = self._apply_command_at_cycle_start(caught_cmd)
            if handled:
                return  # stop/pause handled (pause blocks internally)
            # dismiss/continue path: treat as “action taken”, don’t escalate
            self.state["no_action_streak"] = 0
            self.state["awaiting_action"] = False
            self.state["pending_inputs"] = []
            self._save_state()
            return  # ← Important: stop this cycle cleanly if user responded

        # 6) Auto-open Pause menu after 3 no-action cycles
        if self.state["no_action_streak"] >= 5:
            self._pause_menu(blocking_reason="No response for 3 cycles; pausing until Continue/Stop.",
                             input_names=input_names)
            return

        # Otherwise, we finish this cycle and try again next time

    # ----------------------------- helpers -----------------------------

    # ---------------- shared UPDATES.json liquid ledger helpers ----------------

    def _blank(self, v):
        return v is None or v == "" or v == "null"

    def _as_float_or_none(self, v):
        if self._blank(v):
            return None
        return float(v)

    def _as_list(self, value, n, default=None):
        if isinstance(value, list):
            out = value[:n]
        else:
            out = [value for _ in range(n)]
        while len(out) < n:
            out.append(default)
        return out

    def _load_updates_conf(self):
        import json
        with open(self.updates_path, "r", encoding="utf-8") as f:
            conf = json.load(f)
        return self._ensure_liquid_schema(conf)

    def _save_updates_conf(self, conf):
        import json
        with open(self.updates_path, "w", encoding="utf-8") as f:
            json.dump(conf, f, ensure_ascii=False, indent=2)

    def _ensure_liquid_schema(self, conf):
        n = int(conf.get("real_salt_no", self.num_inputs))

        salt_tank = self._as_list(conf.get("Salt_Tank", self.Salt_Tank), n, 50000)
        water_tank = conf.get("Water_Tank", self.Water_Tank)

        conf.setdefault("warning_margin_uL", 3000)
        conf.setdefault("protocol_refill_margin_uL", 1000)
        conf.setdefault("physical_tube_capacity_uL", 50000)

        req = conf.setdefault("refill_request", {})
        req.setdefault("enabled", False)
        req["salt_uL"] = self._as_list(req.get("salt_uL", [None] * n), n, None)
        req.setdefault("water_uL", None)

        state = conf.setdefault("liquid_state", {})
        state["salt_fill_uL"] = [
            float(x) for x in self._as_list(state.get("salt_fill_uL", salt_tank), n, 50000)
        ]
        state["salt_refill_at_usage_uL"] = [
            float(x) for x in self._as_list(state.get("salt_refill_at_usage_uL", [0] * n), n, 0)
        ]
        state["water_fill_uL"] = float(state.get("water_fill_uL", water_tank))
        state["water_refill_at_usage_uL"] = float(state.get("water_refill_at_usage_uL", 0))
        return conf

    def _current_usage_from_updates(self, conf):
        n = int(conf.get("real_salt_no", self.num_inputs))
        salt_used = [
            float(x) for x in self._as_list(conf.get("input_column_sums", [0] * n), n, 0)
        ]

        # Critical: use the same volume recorded in UPDATES.json.
        # Do not hard-code 1100 or 1200 here.
        cell_vol = float(conf.get("cell_volume_uL", 0))
        cycle_no = int(conf.get("psycho", self.Psycho))

        water_used = max(0.0, cycle_no * cell_vol - sum(salt_used))
        input_names = [f"Input {i + 1}" for i in range(n)] + ["Water"]

        return salt_used + [water_used], input_names

    def _apply_refill_request_if_enabled(self, conf):
        """
        If the user edited UPDATES.json and set refill_request.enabled = true,
        commit the refill into liquid_state and turn the trigger off.
        """
        conf = self._ensure_liquid_schema(conf)
        req = conf["refill_request"]

        if not bool(req.get("enabled", False)):
            return conf, False

        usage, _input_names = self._current_usage_from_updates(conf)
        state = conf["liquid_state"]
        changed = False

        for i, v in enumerate(req.get("salt_uL", [])):
            refill_uL = self._as_float_or_none(v)
            if refill_uL is not None:
                state["salt_fill_uL"][i] = refill_uL
                state["salt_refill_at_usage_uL"][i] = usage[i]
                changed = True

        water_refill_uL = self._as_float_or_none(req.get("water_uL"))
        water_idx = int(conf.get("real_salt_no", self.num_inputs))
        if water_refill_uL is not None:
            state["water_fill_uL"] = water_refill_uL
            state["water_refill_at_usage_uL"] = usage[water_idx]
            changed = True

        req["enabled"] = False
        return conf, changed

    # ---------------------------------------------------------------------------

    def _load_state(self):
        import os, json
        defaults = {
            "last_update_id": None,
            "initiated_sent": False,
            "awaiting_action": False,
            "pending_inputs": [],
            "no_action_streak": 0,
            "thresholds": None,   # list per input
            "bump_counts": None,  # list per input
        }
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in defaults.items():
                    if k not in data:
                        data[k] = v
                return data
        except Exception:
            pass
        return defaults

    def _save_state(self):
        import json
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def _log_update(self, upd):
        import json, time
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "update_id": upd.get("update_id"),
                    "type": "callback_query" if "callback_query" in upd else "message" if "message" in upd else "other",
                    "data": (upd.get("callback_query") or {}).get("data"),
                    "text": (upd.get("message") or {}).get("text"),
                    "date": (upd.get("message") or upd.get("callback_query") or {}).get("date"),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _send_text(self, text):
        import requests
        try:
            requests.post(f"{self.base}/sendMessage",
                          json={"chat_id": self.chat_id, "text": text})
        except Exception as e:
            print(f"[feedback] send_text failed: {e}")

    def _ack_cmd(self, cmd):
        # small, immediate confirmation that we *received* the command
        self._send_text(f"✅ received: {cmd}")

    def _send_warning_with_buttons(self, title, body):
        import requests
        kb = {
            "inline_keyboard": [[
                {"text": "🛑 Stop", "callback_data": "stop"},
                {"text": "⏸ Pause", "callback_data": "pause"},
                {"text": "✔️ Dismiss", "callback_data": "dismiss"},
            ]]
        }
        try:
            requests.post(f"{self.base}/sendMessage",
                          json={"chat_id": self.chat_id, "text": f"{title}\n{body}",
                                "reply_markup": kb})
        except Exception as e:
            print(f"[feedback] send_warning failed: {e}")

    def _send_pause_buttons(self):
        import requests
        kb = {"inline_keyboard": [[
            {"text": "▶ Continue", "callback_data": "continue"},
            {"text": "🛑 Stop", "callback_data": "stop"},
        ]]}
        try:
            requests.post(f"{self.base}/sendMessage",
                          json={"chat_id": self.chat_id, "text": "Paused — choose:",
                                "reply_markup": kb})
        except Exception:
            pass

    def _answer_callback(self, callback_query_id):
        import requests
        try:
            requests.post(f"{self.base}/answerCallbackQuery",
                          json={"callback_query_id": callback_query_id})
        except Exception:
            pass

    def _get_updates_raw(self, offset=None, timeout=0):
        import requests
        params = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            params["offset"] = offset
        r = requests.get(f"{self.base}/getUpdates", params=params, timeout=max(5, timeout + 2))
        try:
            data = r.json()
        except Exception:
            data = {"ok": False, "result": []}
        return data

    def _poll_and_select_best(self):
        """
        Poll once and return the best command seen (stop > pause > dismiss > continue),
        advancing last_update_id. Also logs every update and answers callbacks.
        """
        data = self._get_updates_raw(
            offset=(self.state["last_update_id"] + 1) if self.state["last_update_id"] is not None else None,
            timeout=1
        )
        results = data.get("result", [])
        if not results:
            return None

        priority = {"stop": 3, "pause": 2, "dismiss": 1, "continue": 0}
        best = None

        for upd in results:
            self._log_update(upd)
            self.state["last_update_id"] = upd["update_id"]

            if "callback_query" in upd:
                data = (upd["callback_query"].get("data") or "").strip().lower()
                if data in priority:
                    if (best is None) or (priority[data] > priority[best]):
                        best = data
                # clear the loading spinner
                try:
                    self._answer_callback(upd["callback_query"]["id"])
                except Exception:
                    pass

            elif "message" in upd and "text" in upd["message"]:
                text = (upd["message"]["text"] or "").strip().lower()
                if text in priority:
                    if (best is None) or (priority[text] > priority[best]):
                        best = text

        self._save_state()
        return best

    def _catch_window(self, seconds=30):
        """
        After sending a warning, wait up to `seconds` for a button press.
        Polls repeatedly, returns the best command seen (or None).
        """
        import time
        deadline = time.time() + max(0, seconds)
        best = None
        priority = {"stop": 3, "pause": 2, "dismiss": 1, "continue": 0}
        while time.time() < deadline:
            cmd = self._poll_and_select_best()
            if cmd is not None:
                # choose higher priority if multiple arrive during the window
                if (best is None) or (priority[cmd] > priority[best]):
                    best = cmd
                # early exit if stop or pause (high priority) to act immediately
                if cmd in ("stop", "pause"):
                    break
            time.sleep(2)  # gentle polling cadence
        return best

    def _compute_usage(self, csv_module=None):
        """
        UPDATES.json is now the liquid ledger.

        Returns:
            usage: cumulative salt usage from input_column_sums + virtual Water usage.
            input_names: generic labels used for Telegram messages.
        """
        conf = self._load_updates_conf()
        conf, changed = self._apply_refill_request_if_enabled(conf)
        if changed:
            self._save_updates_conf(conf)

        usage, input_names = self._current_usage_from_updates(conf)
        return usage, input_names

    def _find_flagged_inputs(self, usage):
        """Warn based on remaining liquid in the shared UPDATES liquid_state."""
        conf = self._load_updates_conf()
        state = conf["liquid_state"]
        margin = float(conf.get("warning_margin_uL", 3000))

        fills = list(state["salt_fill_uL"]) + [state["water_fill_uL"]]
        baselines = list(state["salt_refill_at_usage_uL"]) + [state["water_refill_at_usage_uL"]]

        flagged = []
        for i, total in enumerate(usage):
            if i >= len(fills):
                continue
            used_since_refill = max(0.0, float(total) - float(baselines[i]))
            remaining = float(fills[i]) - used_since_refill
            if remaining <= margin:
                flagged.append(i)

        return flagged

    def _criteria_update(self, inputs_idx, input_names):
        """
        Dismiss/Continue now means: user has refilled the flagged tubes.

        This resets the shared UPDATES liquid_state baselines, which is the same
        object used by RoboticProcess for the OT2 aspiration-height logic.

        If refill_request.enabled is true and a value exists, that value becomes
        the new fill amount. Otherwise, we assume the user refilled back to the
        current effective fill amount stored in liquid_state.
        """
        conf = self._load_updates_conf()
        usage, _ = self._current_usage_from_updates(conf)

        req = conf["refill_request"]
        state = conf["liquid_state"]
        request_enabled = bool(req.get("enabled", False))

        msgs = []
        water_idx = self.num_inputs

        for i in inputs_idx:
            if i < self.num_inputs:
                requested = (
                    self._as_float_or_none(req["salt_uL"][i])
                    if request_enabled and i < len(req.get("salt_uL", []))
                    else None
                )
                if requested is not None:
                    state["salt_fill_uL"][i] = requested

                state["salt_refill_at_usage_uL"][i] = usage[i]
                remaining_fill = state["salt_fill_uL"][i]
                msgs.append(f"{input_names[i]} reset at {usage[i]:.1f} µL; active fill = {remaining_fill:.1f} µL")

            elif i == water_idx:
                requested = (
                    self._as_float_or_none(req.get("water_uL"))
                    if request_enabled
                    else None
                )
                if requested is not None:
                    state["water_fill_uL"] = requested

                state["water_refill_at_usage_uL"] = usage[i]
                remaining_fill = state["water_fill_uL"]
                msgs.append(f"{input_names[i]} reset at {usage[i]:.1f} µL; active fill = {remaining_fill:.1f} µL")

        req["enabled"] = False
        self._save_updates_conf(conf)
        self._save_state()

        if msgs:
            self._send_text("Liquid ledger reset:\n" + "\n".join(msgs))


    def _apply_command_at_cycle_start(self, cmd):
        """
        Apply a command that arrived before we re-checked thresholds this cycle.
        Returns True if the command fully handled this call (e.g., stop or pause).
        """
        if cmd == "stop":
            self._send_text("🛑 Stop received. Terminating.")
            raise SystemExit(1)

        elif cmd == "dismiss":
            if self.state.get("awaiting_action") and self.state.get("pending_inputs"):
                import csv as _csv
                usage, input_names = self._compute_usage(_csv)
                self._criteria_update(self.state["pending_inputs"], input_names)
            self.state["awaiting_action"] = False
            self.state["pending_inputs"] = []
            self.state["no_action_streak"] = 0
            self._save_state()
            return False

        elif cmd == "pause":
            import csv as _csv
            _usage, input_names = self._compute_usage(_csv)
            self._pause_menu(blocking_reason="Pause requested.", input_names=input_names)
            return True  # pause menu blocks until Continue/Stop, then returns

        elif cmd == "continue":
            if self.state.get("awaiting_action") and self.state.get("pending_inputs"):
                import csv as _csv
                _usage, input_names = self._compute_usage(_csv)
                self._criteria_update(self.state["pending_inputs"], input_names)
            self.state["awaiting_action"] = False
            self.state["pending_inputs"] = []
            self.state["no_action_streak"] = 0
            self._save_state()
            return False

        return False

    def _pause_menu(self, blocking_reason, input_names):
        """Block here until Continue or Stop. Continue triggers criteria update if pending."""
        self._send_text(f"⏸ Paused. {blocking_reason}\nOptions: Continue / Stop")
        self.state["awaiting_action"] = True
        self._save_state()

        # Present continue/stop buttons
        self._send_pause_buttons()

        # Wait indefinitely for continue or stop
        import time
        while True:
            cmd = self._poll_and_select_best()
            if cmd:
                self._ack_cmd(cmd)
                if cmd == "stop":
                    self._send_text("🛑 Stop received while paused. Terminating.")
                    raise SystemExit(1)
                if cmd == "continue":
                    pend = self.state.get("pending_inputs") or []
                    if pend:
                        self._criteria_update(pend, input_names)
                    self.state["awaiting_action"] = False
                    self.state["pending_inputs"] = []
                    self.state["no_action_streak"] = 0
                    self._save_state()
                    self._send_text("▶ Resuming.")
                    break
            time.sleep(1)




try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

def _pareto_mask_max(y: np.ndarray) -> np.ndarray:
    
    n = y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # any point better-or-equal in all and strictly better in at least one?
        ge = (y >= y[i]).all(axis=1)
        gt = (y > y[i]).any(axis=1)
        dominated_by_someone = ge & gt
        if dominated_by_someone.any():
            keep[i] = False
            continue

        # pruning: drop points dominated by i
        dominated_by_i = (y[i] >= y).all(axis=1) & ((y[i] > y).any(axis=1))
        keep[dominated_by_i] = False
        keep[i] = True
    return keep


def _resolve_output_columns(df: pd.DataFrame,
                            num_inputs: int,
                            num_objectives: int,
                            output_cols: Optional[Sequence[str]] = None) -> List[str]:
    if output_cols is not None:
        output_cols = list(output_cols)
        missing = [c for c in output_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing output columns in CSV: {missing}")
        if len(output_cols) != num_objectives:
            raise ValueError("output_cols length must match num_objectives")
        return output_cols

    cols = list(df.columns)
    if len(cols) < num_inputs + num_objectives:
        raise ValueError(
            f"CSV has {len(cols)} columns but needs at least num_inputs+num_objectives "
            f"({num_inputs}+{num_objectives}={num_inputs+num_objectives})."
        )
    return cols[num_inputs:num_inputs + num_objectives]


def _load_complete_history(csv_path: str,
                           num_inputs: int,
                           num_objectives: int,
                           output_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    out_cols = _resolve_output_columns(df, num_inputs, num_objectives, output_cols=output_cols)

    # keep only rows where ALL objectives exist
    sub = df.copy()
    sub = sub[sub[out_cols].notna().all(axis=1)]

    # if trial_index exists, keep latest per trial_index (handy when resuming / overwriting)
    if "trial_index" in sub.columns:
        sub = sub.sort_values("trial_index").drop_duplicates(subset=["trial_index"], keep="last")

    sub = sub.reset_index(drop=True)
    return sub


def _compute_cutoffs(n_complete: int, psycho: int, update_every: int, panels: int) -> List[int]:
    """
    Desired snapshots at multiples of update_every up to the current psycho.
    We then clamp each cutoff to <= n_complete.
    """
    if psycho < update_every:
        base = 0
    else:
        base = (psycho // update_every) * update_every

    multiples = list(range(update_every, base + 1, update_every))
    multiples = multiples[-panels:]  # last N checkpoints

    # clamp to available completed rows
    cutoffs = [min(m, n_complete) for m in multiples]
    # remove non-positive and duplicates while preserving order
    cleaned = []
    for c in cutoffs:
        if c <= 0:
            continue
        if len(cleaned) == 0 or c != cleaned[-1]:
            cleaned.append(c)
    return cleaned


@dataclass
class TacticalParetoConfig:
    input_file: str
    num_inputs: int
    num_objectives: int
    update_every: int = 3
    panels: int = 4
    maximise: bool = True  # current implementation assumes maximise=True
    output_cols: Optional[Sequence[str]] = None  # override if you prefer named columns
    save_png: bool = False
    save_html: bool = False
    out_dir: Optional[str] = None
    show: bool = True  # if False, it won't pop a live window (still can save PNG/HTML)


class TacticalParetoMonitor:
    """
    Keeps a persistent 2x2 figure alive and updates it every N iterations.
    Supports 2D (2 objectives) and 3D (3 objectives).
    """

    def __init__(self, cfg: TacticalParetoConfig):
        if cfg.num_objectives not in (2, 3):
            raise ValueError("TacticalParetoMonitor currently supports num_objectives = 2 or 3.")
        if not cfg.maximise:
            raise ValueError("Currently implemented for maximise=True. (Easy to extend if needed.)")

        self.cfg = cfg
        self.fig = None
        self.axes = None
        self._last_drawn_cutoffs: List[int] = []

        if self.cfg.show:
            plt.ion()  # interactive mode

    def _ensure_figure(self):
        if self.fig is not None and self.axes is not None:
            return

        self.fig = plt.figure(figsize=(12, 9), dpi=130)
        self.axes = []
        for idx in range(4):
            if self.cfg.num_objectives == 3:
                ax = self.fig.add_subplot(2, 2, idx + 1, projection="3d")
            else:
                ax = self.fig.add_subplot(2, 2, idx + 1)
            self.axes.append(ax)

        self.fig.suptitle("Observed Pareto frontier snapshots (rolling 4 panels)", fontsize=12)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

        if self.cfg.show:
            plt.show(block=False)
            plt.pause(0.1)

        self.fig.suptitle("Observed Pareto frontier snapshots (rolling 4 panels)", fontsize=12)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(self, psycho: int) -> None:
        """
        Call this from the orchestrator with Psycho=K+1 (your iteration counter).
        It will only redraw on multiples of update_every, but it also handles resume
        (first call will draw the latest panels based on existing CSV).
        """
        self._ensure_figure()

        # Load completed history (only rows with all objectives present)
        hist = _load_complete_history(
            self.cfg.input_file,
            num_inputs=self.cfg.num_inputs,
            num_objectives=self.cfg.num_objectives,
            output_cols=self.cfg.output_cols
        )
        n_complete = len(hist)
        if n_complete == 0:
            # nothing to plot yet
            return

        # Decide whether to redraw:
        # - redraw if it's an update checkpoint
        # - OR redraw if this is the first draw (resume scenario)
        at_checkpoint = (psycho % self.cfg.update_every == 0)
        if not at_checkpoint and self._last_drawn_cutoffs:
            return

        cutoffs = _compute_cutoffs(n_complete, psycho, self.cfg.update_every, self.cfg.panels)
        if not cutoffs:
            return

        # Avoid useless redraws if nothing changed
        if cutoffs == self._last_drawn_cutoffs:
            return
        self._last_drawn_cutoffs = cutoffs

        # output columns
        df_full = pd.read_csv(self.cfg.input_file)
        out_cols = _resolve_output_columns(df_full, self.cfg.num_inputs, self.cfg.num_objectives, self.cfg.output_cols)

        # Clear and redraw all 4 panels (hide unused)
        for ax in self.axes:
            ax.clear()

        for k in range(4):
            if k >= len(cutoffs):
                self.axes[k].axis("off")
                continue

            N = cutoffs[k]
            sub = hist.iloc[:N].copy()
            y = sub[out_cols].to_numpy(dtype=float)

            mask = _pareto_mask_max(y)
            pareto = sub.loc[mask].copy()

            ax = self.axes[k]

            if self.cfg.num_objectives == 2:
                x0, x1 = out_cols[0], out_cols[1]
                ax.scatter(sub[x0], sub[x1], s=18, alpha=0.45, label="All")
                ax.scatter(pareto[x0], pareto[x1], s=34, alpha=0.9, label="Pareto")

                ax.set_xlabel(x0)
                ax.set_ylabel(x1)
            else:
                x0, x1, x2 = out_cols[0], out_cols[1], out_cols[2]
                ax.scatter(sub[x0], sub[x1], sub[x2], s=18, alpha=0.45, label="All")
                ax.scatter(pareto[x0], pareto[x1], pareto[x2], s=34, alpha=0.9, label="Pareto")

                ax.set_xlabel(x0)
                ax.set_ylabel(x1)
                ax.set_zlabel(x2)

            ax.set_title(f"0 --> {N}  (n={len(sub)}, front={len(pareto)})", fontsize=10)
            ax.legend(frameon=False, fontsize=8)

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Save outputs (optional)
        out_dir = self.cfg.out_dir or (os.path.dirname(self.cfg.input_file) or ".")
        stem = os.path.splitext(os.path.basename(self.cfg.input_file))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")

        if self.cfg.save_png:
            png_path = os.path.join(out_dir, f"{stem}_pareto_panel_{ts}.png")
            self.fig.savefig(png_path, dpi=250, bbox_inches="tight")

        if self.cfg.save_html and _HAS_PLOTLY:
            # One HTML showing the CURRENT frontier only (latest snapshot)
            latest = hist.copy()
            y_latest = latest[out_cols].to_numpy(dtype=float)
            mask_latest = _pareto_mask_max(y_latest)
            pareto_latest = latest.loc[mask_latest].copy()

            fig = go.Figure()
            if self.cfg.num_objectives == 2:
                x0, x1 = out_cols[0], out_cols[1]
                fig.add_trace(go.Scatter(
                    x=latest[x0], y=latest[x1], mode="markers",
                    name="All", opacity=0.45
                ))
                fig.add_trace(go.Scatter(
                    x=pareto_latest[x0], y=pareto_latest[x1], mode="markers",
                    name="Pareto", opacity=0.9
                ))
                fig.update_layout(xaxis_title=x0, yaxis_title=x1, template="plotly_white")
            else:
                x0, x1, x2 = out_cols[0], out_cols[1], out_cols[2]
                fig.add_trace(go.Scatter3d(
                    x=latest[x0], y=latest[x1], z=latest[x2],
                    mode="markers", name="All", opacity=0.45
                ))
                fig.add_trace(go.Scatter3d(
                    x=pareto_latest[x0], y=pareto_latest[x1], z=pareto_latest[x2],
                    mode="markers", name="Pareto", opacity=0.9
                ))
                fig.update_layout(template="plotly_white")

            html_path = os.path.join(out_dir, f"{stem}_pareto_latest_{ts}.html")
            pio.write_html(fig, html_path, include_plotlyjs="cdn", full_html=True)

        # Live refresh
        if self.cfg.show:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)


# -------------------------------------------------------------------
# Public NIMSOS-style entry point + internal singleton cache
# -------------------------------------------------------------------

_PARETO_MONITORS: Dict[str, TacticalParetoMonitor] = {}


def tactical_pareto(input_file: str,
                    num_inputs: int,
                    num_objectives: int,
                    Psycho: int,
                    update_every: int = 3,
                    panels: int = 4,
                    output_cols: Optional[Sequence[str]] = None,
                    save_png: bool = False,
                    save_html: bool = False,
                    out_dir: Optional[str] = None,
                    show: bool = True) -> None:
    """
    Call this from the orchestrator loop.
    Keeps a persistent figure per input_file path.
    """
    key = os.path.abspath(input_file)
    mon = _PARETO_MONITORS.get(key)

    if mon is None:
        cfg = TacticalParetoConfig(
            input_file=input_file,
            num_inputs=num_inputs,
            num_objectives=num_objectives,
            update_every=update_every,
            panels=panels,
            output_cols=output_cols,
            save_png=save_png,
            save_html=save_html,
            out_dir=out_dir,
            show=show,
        )
        mon = TacticalParetoMonitor(cfg)
        _PARETO_MONITORS[key] = mon

    mon.update(psycho=Psycho)