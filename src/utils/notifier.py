import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()


def send_spill_alert(spill_count, locations_affected):
    """Sends a system-level summary email when spills are detected."""
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
    Affected Locations: {', '.join(locations_affected)}
    Timestamp: {os.popen('date').read()}
    Please check the latest dashboard visualization for details.
    """

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

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
    Turns a metrics.classify_event result into one plain-English sentence for
    the alert email. Handles all three verdicts (diagnosed, possible_new_type,
    undetermined) and falls back gracefully if classification is None.
    """
    if classification is None:
        return "Spill type was not classified for this anomaly."

    verdict = classification.get("verdict")

    if verdict == "diagnosed":
        named = classification.get("named_type", "unknown")
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

    # undetermined, or any unexpected verdict
    candidates = classification.get("top_candidates") or []
    hint = ""
    if candidates:
        hint = f" The leading candidate was {candidates[0]}, but this is a hint only."
    return (
        "A spill type could not be determined. The sensors that separate "
        "pollutant types (dissolved oxygen, pH, floating conductivity) were not "
        "reporting at this site." + hint
    )


def send_anomaly_alert(location, score, threshold, event_time, classification=None):
    """
    Sends one alert email for a single detected anomaly event.

    classification is the metrics.classify_event dict for this event, or None
    if it wasn't classified. Credential handling mirrors send_spill_alert.
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

    subject = f" ALERT: Anomaly Detected at {location} in Strawberry Creek"
    body = f"""
    The SCMG Anomaly Detection System has detected an anomaly.
    Location: {location}
    Time: {time_str}
    Anomaly score: {score:.4f} (threshold {threshold:.4f})

    {diagnosis}

    This detection is based on a sudden deviation in conductivity relative to
    the model's prediction of normal creek behavior at this location. Please
    check the latest dashboard visualization for details.
    """

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

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
    Sends one detailed email per event and one system summary for the full batch.

    Each event dict needs: location, score, threshold, event_time, and
    optionally classification (from metrics.classify_event, or None).
    """
    if not events:
        print("No anomaly events to alert on.")
        return

    for ev in events:
        send_anomaly_alert(
            location=ev.get("location", "unknown"),
            score=ev.get("score", float("nan")),
            threshold=ev.get("threshold", float("nan")),
            event_time=ev.get("event_time", datetime.now(timezone.utc)),
            classification=ev.get("classification"),
        )

    locations = sorted({ev.get("location", "unknown") for ev in events})
    send_spill_alert(len(events), locations)
    