import os
import smtplib
from datetime import UTC, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()


def send_spill_alert(spill_count, locations_affected):
    """Send a system-level summary email when spills are detected."""
    sender = os.getenv("ALERT_EMAIL_SENDER")
    password = os.getenv("ALERT_EMAIL_PASSWORD")
    receiver = os.getenv("ALERT_EMAIL_RECEIVER")

    if not all([sender, password, receiver]):
        print("Alert skipped: Email credentials missing in .env")
        return

    subject = f" ALERT: {spill_count} Potential Spill(s) Detected in Strawberry Creek"
    body = f"""
    The SCMG Anomaly Detection System has identified potential spills.
    Count: {spill_count}
    Affected Locations: {", ".join(locations_affected)}
    Timestamp: {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")}
    Please check the latest dashboard visualization for details.
    """

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT")))
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print("Spill alert email sent successfully.")
    except Exception as e:
        print(f"Failed to send email alert: {e}")


def _diagnosis_line(classification):
    """
    Turn one Trial Bed account into a plain-English sentence for the email.

    Handles every verdict support_modules.trial_bed.classify can return, and
    falls back to saying so when nothing classified the event at all.
    """
    if classification is None:
        return "Spill type was not classified for this anomaly."

    verdict = classification.get("verdict")

    if verdict == "diagnosed":
        named = classification.get("cause", "unknown")
        top = classification["ranked"][0]
        return (
            f"Likely type: {named}. This matched {top['agreements']} of "
            f"{top['comparable']} available water quality parameters. Treat this "
            f"as a lead rather than a confirmation, since confidence depends on "
            f"which sensors were reporting."
        )

    if verdict == "possible_new_type":
        return (
            "The parameter changes did not match any known spill signature. "
            "This may be a new or unclassified event and is worth a closer look."
        )

    # cannot_evaluate, nothing_to_explain, or any unexpected verdict
    candidates = classification.get("top_candidates") or []
    hint = ""
    if candidates:
        hint = f" The leading candidate was {candidates[0]}, but this is a hint only."
    return (
        "A spill type could not be determined. The sensors that separate "
        "pollutant types (dissolved oxygen, pH, floating conductivity) were not "
        "reporting at this site." + hint
    )


# Two alerts for the same site otherwise look identical apart from the numbers,
# so each rule gets its own label and its own sentence.
_RULE_LABELS = {
    "forecast_residual": "forecast residual",
    "level_shift": "level shift",
}

_RULE_LINES = {
    "forecast_residual": (
        "Conductivity kept landing away from what the model expected, step after "
        "step. That is what an onset or a ramp looks like."
    ),
    "level_shift": (
        "Conductivity sat well off this site's normal level and stayed there. The "
        "model stops being surprised by a step within a timestep or two, so this "
        "is the rule that catches one that persists."
    ),
}

_GENERIC_RULE_LINE = (
    "This detection is based on a sudden deviation in conductivity relative to "
    "the model's prediction of normal creek behavior at this location."
)


def _rule_line(rule):
    """Plain sentence for whichever rule fired, or the generic one if unnamed."""
    return _RULE_LINES.get(rule, _GENERIC_RULE_LINE)


def send_anomaly_alert(location, score, threshold, event_time, classification=None, rule=None):
    """
    Send one alert email for a single detected anomaly event.

    classification is the support_modules.trial_bed account for this event, or
    None if nothing classified it. rule is which detection rule fired, so a site
    that trips both gets two distinguishable emails. Credential handling mirrors
    send_spill_alert.
    """
    sender = os.getenv("ALERT_EMAIL_SENDER")
    password = os.getenv("ALERT_EMAIL_PASSWORD")
    receiver = os.getenv("ALERT_EMAIL_RECEIVER")

    if not all([sender, password, receiver]):
        print("Alert skipped: Email credentials missing in .env")
        return

    diagnosis = _diagnosis_line(classification)

    if isinstance(event_time, datetime):
        time_str = event_time.isoformat()
    else:
        time_str = str(event_time)

    tag = f" ({_RULE_LABELS[rule]})" if rule in _RULE_LABELS else ""
    subject = f" ALERT: Anomaly Detected at {location}{tag} in Strawberry Creek"
    body = f"""
    The SCMG Anomaly Detection System has detected an anomaly.
    Location: {location}
    Time: {time_str}
    Rule fired: {_RULE_LABELS.get(rule, "unspecified")}
    Anomaly score: {score:.4f} (threshold {threshold:.4f})

    {diagnosis}

    {_rule_line(rule)}

    Check the latest dashboard visualization for details.
    """

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT")))
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print(f"Anomaly alert email sent successfully for {location}.")
    except Exception as e:
        print(f"Failed to send anomaly alert email: {e}")


def fire_anomaly_alerts(events):
    """
    Send one detailed email per event and one system summary for the full batch.

    Each event dict needs: location, score, threshold, event_time, and
    optionally rule (which detection rule fired) and classification (one
    support_modules.trial_bed account, or None).
    """
    if not events:
        print("No anomaly events to alert on.")
        return

    for ev in events:
        send_anomaly_alert(
            location=ev.get("location", "unknown"),
            score=ev.get("score", float("nan")),
            threshold=ev.get("threshold", float("nan")),
            event_time=ev.get("event_time", datetime.now(UTC)),
            classification=ev.get("classification"),
            rule=ev.get("rule"),
        )

    locations = sorted({ev.get("location", "unknown") for ev in events})
    send_spill_alert(len(events), locations)
