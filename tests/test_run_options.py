"""
main.py's per-run options, the notifier's delivery report, and config's
settings overlay. All three default to how things always ran, and the
dashboard leans on every one of them.
"""

import argparse
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main
from strawberrywatch import config
from strawberrywatch.anomalies import anomaly_detector as ad
from strawberrywatch.utils import notifier

ROOT = Path(__file__).resolve().parents[1]


def _event(site, rule="level_shift", classification=None):
    return {
        "location": site,
        "rule": rule,
        "score": 5.0,
        "threshold": 4.0,
        "event_time": pd.Timestamp("2026-10-10T12:00", tz="UTC"),
        "classification": classification,
    }


def test_operating_point_reads_each_key_as_its_own_type():
    assert main._operating_point_item("level_shift_k=5") == ("level_shift_k", 5.0)
    key, value = main._operating_point_item("min_timesteps_over=4")
    assert (key, value) == ("min_timesteps_over", 4) and isinstance(value, int)


@pytest.mark.parametrize(
    "text",
    [
        "level_shiftk=5",
        "level_shift_k",
        "min_timesteps_over=2.5",
        "level_shift_k=-1",
        "level_shift_k=nan",
        "level_shift_k=lots",
    ],
)
def test_operating_point_refuses_what_detect_anomalies_would_ignore_or_choke_on(text):
    with pytest.raises(argparse.ArgumentTypeError):
        main._operating_point_item(text)


def test_every_operating_point_key_is_one_detect_anomalies_reads():
    # detect_anomalies merges the override over its own dict and ignores any key
    # it doesn't know. A key main.py accepts and it doesn't would have no effect.
    source = inspect.getsource(ad.detect_anomalies)
    for key in main._OPERATING_POINT:
        assert f'"{key}":' in source, key


def test_level_shift_events_report_the_k_they_were_held_to():
    node_results = {
        "oxford": {
            "judged": True,
            "flagged": True,
            "rules_fired": ["level_shift"],
            "rule1": {"peak_deviation": 1.0},
            "rule2": {"peak_deviation": 7.0, "over_timesteps": np.array([False, True])},
            "n_real": 40,
        }
    }
    events = main._anomaly_events(
        node_results, ["t0", "t1"], {"node_thresholds": {}}, {}, ["oxford"], level_shift_k=6.5
    )
    assert events[0]["threshold"] == 6.5
    assert events[0]["event_time"] == "t1"


def test_muted_sites_are_kept_but_not_sent():
    sent = []
    events = [_event("oxford"), _event("north_fork_0")]
    main._dispatch_alerts(
        events, alert_sites=["north_fork_0"], send=lambda evs: sent.extend(evs) or [True] * len(evs)
    )
    assert [ev["location"] for ev in sent] == ["north_fork_0"]
    assert [ev["status"] for ev in events] == ["muted", "emailed"]


def test_no_alert_sites_means_every_site():
    events = [_event("oxford")]
    main._dispatch_alerts(events, send=lambda evs: [True])
    assert events[0]["status"] == "emailed"


def test_a_dry_run_sends_nothing():
    events = [_event("oxford"), _event("north_fork_0")]
    main._dispatch_alerts(
        events,
        alert_sites=["oxford"],
        dry_run=True,
        send=lambda evs: pytest.fail("sent on a dry run"),
    )
    assert [ev["status"] for ev in events] == ["dry_run", "muted"]


def test_status_comes_from_what_the_notifier_says_went():
    events = [_event("oxford"), _event("north_fork_0")]
    main._dispatch_alerts(events, send=lambda evs: [True, False])
    assert [ev["status"] for ev in events] == ["emailed", "not_emailed"]


def test_a_notifier_that_raises_leaves_everything_unsent(capsys):
    def broken(evs):
        raise RuntimeError("smtp down")

    events = [_event("oxford")]
    main._dispatch_alerts(events, send=broken)
    assert events[0]["status"] == "not_emailed"
    assert "Alerting failed: smtp down" in capsys.readouterr().out


def test_results_file_is_strict_json_with_nan_as_null(tmp_path):
    node_results = {
        "oxford": {
            "judged": True,
            "flagged": True,
            "rules_fired": ["level_shift"],
            "n_real": np.int64(40),
            "rule1": {
                "peak_deviation": float("nan"),
                "longest_run": 0,
                "n_over_threshold": np.int64(0),
                "over_timesteps": np.zeros(3, dtype=bool),
            },
            "rule2": {
                "peak_deviation": np.float32(5.0),
                "longest_run": 3,
                "n_over_threshold": 3,
                "over_timesteps": np.ones(3, dtype=bool),
            },
            "over_timesteps": np.ones(3, dtype=bool),
        },
        "north_fork_0": {"judged": False, "reason": "insufficient_real_data", "n_real": 2},
    }
    accounts = {
        "trial_bed": [
            {"verdict": "cannot_evaluate", "cause": None, "confidence": "none", "explanation": "x"},
            {"verdict": "nothing_to_explain"},
        ]
    }
    classification = {"verdict": "diagnosed", "cause": "sewage", "explanation": "Likely sewage"}
    events = [_event("oxford", classification=classification) | {"status": "dry_run"}]
    path = tmp_path / "nested" / "results.json"
    header = {"model": "dusk_crayfish", "window": [np.datetime64("2026-10-10T12:00")]}

    main._write_results(
        str(path), header, node_results, accounts, ["oxford", "north_fork_0"], events
    )

    data = json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"{c} in results"))
    oxford = data["sites"]["oxford"]
    assert oxford["rule1"]["peak_deviation"] is None
    assert oxford["rule2"]["peak_deviation"] == 5.0
    assert "over_timesteps" not in oxford["rule1"] and "over_timesteps" not in oxford
    assert data["sites"]["north_fork_0"] == {
        "judged": False,
        "n_real": 2,
        "reason": "insufficient_real_data",
    }
    assert data["explanations"] == {
        "trial_bed": {
            "oxford": {
                "verdict": "cannot_evaluate",
                "cause": None,
                "confidence": "none",
                "explanation": "x",
            }
        }
    }
    alert = data["alerts"][0]
    assert (alert["status"], alert["diagnosis_cause"]) == ("dry_run", "sewage")
    assert alert["event_time"].startswith("2026-10-10T12:00")
    assert data["window"] == ["2026-10-10T12:00:00"]


def test_main_refuses_unknown_operating_point_keys_before_downloading(monkeypatch, capsys):
    monkeypatch.setattr(main, "load_and_preprocess_data", lambda **kw: pytest.fail("downloaded"))
    with pytest.raises(SystemExit):
        main.main(mode="inference", operating_point={"level_shiftk": 5})
    assert "unknown operating point keys" in capsys.readouterr().out


def test_main_refuses_sites_off_the_roster_before_downloading(monkeypatch, capsys):
    monkeypatch.setattr(main, "load_and_preprocess_data", lambda **kw: pytest.fail("downloaded"))
    with pytest.raises(SystemExit):
        main.main(mode="inference", alert_sites=["atlantis"])
    assert "not on the roster" in capsys.readouterr().out


def test_the_command_line_takes_the_new_flags_and_rejects_a_bad_one():
    help_text = subprocess.run(
        [sys.executable, "main.py", "--help"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    for flag in ("--operating-point", "--alert-sites", "--dry-run", "--results-json"):
        assert flag in help_text
    bad = subprocess.run(
        [sys.executable, "main.py", "--operating-point", "bogus=1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2 and "bogus=1" in bad.stderr


def test_receivers_split_on_commas(monkeypatch):
    monkeypatch.setenv("ALERT_EMAIL_RECEIVER", " a@x.org, b@y.org ,, ")
    assert notifier._receivers() == ["a@x.org", "b@y.org"]


def test_notifier_reports_which_emails_went_and_to_everyone(monkeypatch):
    sent = []

    class FakeSMTP:
        def __init__(self, host, port):
            pass

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def sendmail(self, sender, receivers, message):
            sent.append(receivers)

        def quit(self):
            pass

    monkeypatch.setattr(notifier.smtplib, "SMTP", FakeSMTP)
    env = {
        "ALERT_EMAIL_SENDER": "s@x.org",
        "ALERT_EMAIL_PASSWORD": "pw",
        "ALERT_EMAIL_RECEIVER": "a@x.org, b@y.org",
        "SMTP_SERVER": "smtp.example.org",
        "SMTP_PORT": "587",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert notifier.fire_anomaly_alerts([_event("oxford")]) == [True]
    # one email for the event, one summary, both to the whole list
    assert sent == [["a@x.org", "b@y.org"], ["a@x.org", "b@y.org"]]


def test_notifier_without_anyone_to_send_to_says_nothing_went(monkeypatch):
    monkeypatch.setenv("ALERT_EMAIL_RECEIVER", "")
    assert notifier.fire_anomaly_alerts([_event("oxford")]) == [False]


def test_overrides_lay_over_settings_a_section_at_a_time():
    base = {
        "detection": {"rain_multiplier": 2.0, "threshold_percentile": 99},
        "training": {"epochs": 10},
    }
    merged = config._layered(base, {"detection": {"rain_multiplier": 3.0}})
    assert merged == {
        "detection": {"rain_multiplier": 3.0, "threshold_percentile": 99},
        "training": {"epochs": 10},
    }
    assert base["detection"]["rain_multiplier"] == 2.0


def test_no_override_path_means_no_overrides():
    assert config._read_overrides(None) == {}


def test_a_missing_override_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        config._read_overrides(str(tmp_path / "nope.yaml"))


@pytest.mark.parametrize("text", ["- a list\n", "detection: 3\n"])
def test_an_override_file_the_wrong_shape_raises(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        config._layered({}, config._read_overrides(str(path)))


def test_config_reads_the_override_file_the_environment_names(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("training:\n  epochs: 3\ndetection:\n  rain_multiplier: 2.5\n")
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from strawberrywatch.config import Config as C; "
            "print(C.EPOCHS, C.RAIN_THRESHOLD_MULTIPLIER, C.BATCH_SIZE, C.SETTINGS_OVERRIDES)",
        ],
        env={**os.environ, config.SETTINGS_ENV_VAR: str(path)},
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    epochs, rain, batch, source = out.stdout.split()
    assert (epochs, rain, source) == ("3", "2.5", str(path))
    # a key the file leaves out keeps settings.yaml's value
    assert int(batch) == config.Config.BATCH_SIZE
