# Reference code to paste into Night Heron's email_alerts.py.
# NOT runnable inside StrawberryWatch. Assumes the Night Heron environment:
# Django, base.models.AlertEvent, EMAIL_USER/EMAIL_PASS/FROM_EMAIL credentials,
# the _send_sms function, and the standard email_alerts.py imports are all in scope.
# Paste everything below into the alert delivery section.


def _diagnosis_line(classification):
    """
    Turns a classify_event result into the spill-type sentence for an alert email.
    Takes a plain dict to avoid importing StrawberryWatch code (keeps the repos decoupled).
    Handles all three verdicts so the email never overstates what the classifier knows.
    """
    if classification is None:
        return "Spill type was not classified for this anomaly."

    verdict = classification.get("verdict")

    if verdict == "diagnosed":
        named = classification.get("named_type", "unknown")
        top = classification["ranked"][0]
        return (
            f"Likely type: {named}. Matched {top['agreements']} of "
            f"{top['comparable']} available water quality parameters. Treat as a "
            f"lead, not a confirmation; confidence depends on which sensors "
            f"were reporting."
        )

    if verdict == "possible_new_type":
        return (
            "The parameter changes did not match any known spill signature. "
            "This may be a new or unclassified event and is worth a closer look."
        )

    candidates = classification.get("top_candidates") or []
    hint = ""
    if candidates:
        hint = f" Leading candidate was {candidates[0]}, but this is a hint only."
    return (
        "Spill type could not be determined. The discriminating sensors "
        "(dissolved oxygen, pH, floating conductivity) were not reporting at "
        "this site." + hint
    )


def _send_anomaly_email(rcpts, site, score, threshold, event_time, classification=None):
    """
    Sends a GNN anomaly alert email using the same Elastic Email SMTP and credentials as _send_email.
    Returns the same status strings so the caller can handle success/failure the same way.
    """
    if not rcpts:
        logger.info(f"No email recipients for anomaly alert at {site}. Skipping email.")
        return "skipped_no_recipients"
    if not EMAIL_USER or not EMAIL_PASS or not FROM_EMAIL:
        logger.warning("Email credentials not set. Cannot send anomaly email alerts.")
        return "failure_no_creds"

    diagnosis = _diagnosis_line(classification)

    msg = EmailMessage()
    msg["From"], msg["To"] = FROM_EMAIL, ", ".join(rcpts)
    msg["Subject"] = f"Creek Anomaly Alert {site}: score {score:.3f}"
    content = (
        f"An anomaly was detected at {site}.\n\n"
        f"Time: {event_time.isoformat()}\n"
        f"Anomaly score: {score:.4f} (threshold {threshold:.4f})\n\n"
        f"{diagnosis}\n\n"
        f"Detected by the conductivity anomaly model, which flags sudden "
        f"deviations from the predicted normal behavior of the creek at this site."
    )
    msg.set_content(content)

    try:
        with smtplib.SMTP("smtp.elasticemail.com", 2525, timeout=25) as s:
            s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
        logger.info(f"Anomaly email alert sent for {site} to {len(rcpts)} recipients.")
        return "success"
    except Exception as e:
        logger.error(f"Failed to send anomaly email alert for {site}: {e}", exc_info=True)
        return "failure_send_error"


def fire_anomaly_alert_task(site, score, threshold, event_time_iso, emails, phones,
                            classification=None, created_by_user_id=None, group_obj_id=None):
    """
    Worker task for a GNN anomaly alert, mirroring fire_alerts_task. Sends email,
    optionally SMS, and logs an AlertEvent so anomaly alerts share the same audit trail
    as threshold alerts. event_time comes in as an ISO string because spawned worker
    arguments have to be picklable.
    """
    try:
        django.setup()
        from django.contrib.auth.models import User, Group
        from base.models import AlertEvent
    except RuntimeError as e:
        logger.debug(f"Django already set up in worker process: {e}")
    except Exception as e:
        logger.error(f"Error setting up Django in anomaly alert worker: {e}", exc_info=True)
        return

    event_time = datetime.fromisoformat(event_time_iso)

    detailed_email_status = "pending"
    try:
        if emails:
            detailed_email_status = _send_anomaly_email(emails, site, score, threshold, event_time, classification)
        else:
            detailed_email_status = "skipped_no_recipients"
    except Exception as e_email:
        logger.error(f"Unhandled error sending anomaly email for {site}: {e_email}", exc_info=True)
        detailed_email_status = "failure_exception_calling"
    db_email_status = "success" if detailed_email_status == "success" else "failure"

    # _send_sms expects a numeric Series, so wrap the score in one under the
    # conductivity label since that's what the detector scores on.
    detailed_sms_status = "pending"
    try:
        if phones:
            detailed_sms_status = _send_sms(pd.Series([score]), phones, site, "conductivity")
        else:
            detailed_sms_status = "skipped_no_recipients"
    except Exception as e_sms:
        logger.error(f"Unhandled error sending anomaly SMS for {site}: {e_sms}", exc_info=True)
        detailed_sms_status = "failure_exception_calling"
    db_sms_status = "success" if detailed_sms_status == "success" else "failure"

    named_type = classification.get("named_type") if classification else None
    verdict = classification.get("verdict") if classification else "unclassified"
    notes = f"GNN anomaly alert. Verdict: {verdict}."
    if named_type:
        notes += f" Likely type: {named_type}."

    current_worker_user = None
    current_worker_group = None
    try:
        if created_by_user_id:
            current_worker_user = User.objects.get(id=created_by_user_id)
    except Exception as e:
        logger.error(f"Anomaly worker: error fetching user {created_by_user_id}: {e}", exc_info=True)
    try:
        if group_obj_id:
            current_worker_group = Group.objects.get(id=group_obj_id)
    except Exception as e:
        logger.error(f"Anomaly worker: error fetching group {group_obj_id}: {e}", exc_info=True)

    try:
        AlertEvent.log_event(
            site_code=site,
            sensor_type="conductivity",
            alert_type="gnn_anomaly",
            trigger_value=score,
            rain_pause_applied=False,
            email_status=db_email_status,
            sms_status=db_sms_status,
            notes=notes,
            created_by_user=current_worker_user,
            group_obj=current_worker_group,
        )
    except Exception as e_log:
        logger.error(f"CRITICAL: failed to log GNN anomaly AlertEvent for {site}: {e_log}", exc_info=True)


# How a StrawberryWatch anomaly actually reaches this daemon.
# Nothing above gets called until something invokes fire_anomaly_alert_task.
# The daemon's main loop only iterates Django rule objects, so it never touches
# the GNN. Two options for wiring this up when the systems actually merge:
#
# Option A (recommended, fits the daemon's table-polling pattern):
#   StrawberryWatch writes each flagged anomaly as a row into a shared MySQL table
#   (gnn_anomalies) with site, score, threshold, event_time, a classification JSON
#   blob, and a sent flag. Each daemon cycle reads unsent rows and submits them:
#
#     for row in unsent_gnn_anomaly_rows():
#         ALERT_POOL.apply_async(
#             fire_anomaly_alert_task,
#             args=(row["site"], row["score"], row["threshold"],
#                   row["event_time"].isoformat(),
#                   recipients_for(row["site"]), phones_for(row["site"]),
#                   json.loads(row["classification"]) if row["classification"] else None,
#                   system_user.id if system_user else None,
#                   alerts_group.id if alerts_group else None),
#         )
#         mark_row_sent(row["id"])
#
#   Keeps the daemon as the single owner of delivery and the AlertEvent log.
#   StrawberryWatch never needs Night Heron's credentials.
#
# Option B (simpler, probably not worth it):
#   StrawberryWatch imports and calls _send_anomaly_email directly with Night
#   Heron's credentials. Couples the repos and splits alert ownership, so A
#   is likely the better call.
