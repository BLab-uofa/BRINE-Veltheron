@echo off
python replay_chariot.py --file data\demo_candidates.csv
python campaign_report.py --file data\demo_candidates.csv --decision-log outputs\chariot_replay.csv
pause
