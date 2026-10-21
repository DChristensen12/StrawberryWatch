"""
Flask-WTF forms. These only validate, templates/macros.html draws them. Site
and model choices are built per request. An unknown name fails validation
instead of getting stored.
"""

from __future__ import annotations

import math

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    RadioField,
    SelectField,
    SelectMultipleField,
    StringField,
    TextAreaField,
)
from wtforms.validators import DataRequired, Length, Optional, ValidationError

from strawberrywatch.watch import settings as ws


def _strip(text):
    return text.strip() if text else text


def number_between(lo, hi, what):
    def check(_form, field):
        try:
            value = float(str(field.data).strip())
        except (TypeError, ValueError):
            raise ValidationError(f"{what} must be a number, like 1.0") from None
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValidationError(f"{what} must be between {lo:g} and {hi:g}")

    return check


class NoteForm(FlaskForm):
    note = TextAreaField(
        "Add a note",
        description="For example, what you found on site.",
        filters=[lambda text: text.strip() if text else text],
        validators=[
            DataRequired(message="Enter a note"),
            Length(max=2000, message="Note must be 2,000 characters or less"),
        ],
    )


class RecipientForm(FlaskForm):
    email = StringField(
        "Email address",
        filters=[_strip],
        validators=[DataRequired(message="Enter an email address"), Length(max=254)],
    )
    name = StringField(
        "Name",
        description="Optional.",
        filters=[_strip],
        validators=[Optional(), Length(max=100, message="Name must be 100 characters or less")],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Shown under the list after saving, if the domain couldn't be looked up
        self.note = ""

    def validate_email(self, field):
        from strawberrywatch.dashboard import mail

        try:
            field.data, self.note = mail.check_address(field.data)
        except mail.MailError as exc:
            raise ValidationError(str(exc)) from None

    def validate_name(self, field):
        from strawberrywatch.dashboard import mail

        try:
            mail.check_name(field.data)
        except mail.MailError as exc:
            raise ValidationError(str(exc)) from None


def labelled_recipient_form(labels):
    """The website's add form: the same fields, plus the one label each person gets."""

    class Form(RecipientForm):
        label = SelectField(
            "Label",
            description="Labels are only for telling people apart. They don't give any access.",
            choices=[(str(lb["id"]), lb["name"]) for lb in labels],
            validators=[DataRequired(message="Choose a label")],
        )

    return Form


class LabelForm(FlaskForm):
    name = StringField(
        "Label name",
        filters=[_strip],
        validators=[
            DataRequired(message="Enter a label name"),
            Length(max=50, message="Label name must be 50 characters or less"),
        ],
    )


def settings_form(sites, labels, models):
    """Settings for everyone: which sites, how many alerts, rain, and which models are on."""

    class Form(FlaskForm):
        watch = SelectMultipleField(
            "Sites with alerts on",
            choices=[(s, labels.get(s, s)) for s in sites],
            description="Unticked sites are still checked, but don't raise alerts.",
            validators=[Optional()],
        )
        alerts = RadioField(
            "Alert sensitivity",
            choices=[(key, f"{label}. {sentence}") for key, label, _, sentence in ws.PRESETS],
            validators=[Optional()],
        )
        rain = BooleanField(
            "Raise the alert line during and after rain",
            description="Rain changes most readings, so without this, storms set off alerts. "
            "Leave it on unless you're testing.",
        )

    for model in models:
        setattr(
            Form,
            f"model_{model.name}",
            RadioField(
                model.title,
                choices=[("on", "On"), ("off", "Off")],
                validators=[Optional()],
            ),
        )
    return Form


def advanced_form(models, support_options):
    """Advanced settings: each model's own values, the repeat window."""

    class Form(FlaskForm):
        repeat_hours = StringField(
            "Ignore repeat alerts for",
            description=ws.range_hint("repeat_hours", "hours"),
            validators=[number_between(*ws.REPEAT_RANGE, "Ignore repeat alerts for")],
        )

    lo, hi = ws.SCALE_RANGE
    for model in models:
        setattr(
            Form,
            f"model_{model.name}",
            RadioField(
                model.title,
                choices=[("on", "On"), ("off", "Off")],
                validators=[Optional()],
            ),
        )
        if model.sensitivity == "rate":
            _rate_fields(Form, model)
        for rule in model.rules if model.sensitivity == "scale" else ():
            label = ws.RULE_WORDS.get(rule, (rule, ""))[0]
            setattr(
                Form,
                f"scale_{model.name}_{rule}",
                StringField(
                    label,
                    description=f"{ws.RULE_WORDS.get(rule, ('', ''))[1]} "
                    f"{ws.range_hint('scale', 'times the tested value')}",
                    validators=[number_between(lo, hi, f"{model.title} {label.lower()}")],
                ),
            )
        setattr(
            Form,
            f"rain_{model.name}",
            StringField(
                "Rain multiplier",
                description="How much the alert line rises during rain. 1 means it doesn't rise. "
                + ws.range_hint("rain_multiplier"),
                validators=[number_between(*ws.RAIN_RANGE, f"{model.title} rain multiplier")],
            ),
        )
        if "leave_out" in model.site_choice:
            setattr(
                Form,
                f"leave_{model.name}",
                SelectMultipleField(
                    "Sites to leave out",
                    choices=[(s, ws.site_name(s)) for s in model.sites],
                    description="This model won't use or check a left-out site's data.",
                    validators=[Optional()],
                ),
            )
        setattr(
            Form,
            f"support_{model.name}",
            SelectMultipleField(
                "Support modules",
                choices=[(o["name"], o["title"]) for o in support_options[model.name]],
                validators=[Optional()],
            ),
        )
    return Form


def _rate_fields(Form, model):
    """Calibrated models get a five-stop slider per test, plus one typed threshold."""
    stops = [str(i) for i in range(len(ws.stops()))]

    def a_stop(_form, field):
        if str(field.data).strip() not in stops:
            raise ValidationError(f"Choose one of the {len(stops)} stops")

    for rule in model.rules:
        words = ws.RULE_WORDS.get(rule, (rule, ""))
        setattr(
            Form,
            f"rate_{model.name}_{rule}",
            StringField(words[0], description=words[1], validators=[a_stop]),
        )
    keys = [f"{key}.{rule}" for key in sorted(model.tails) for rule in model.rules]
    setattr(
        Form,
        f"typed_key_{model.name}",
        SelectField(
            "Sensor and test",
            choices=[("", "Choose one")] + [(k, ws.sensor_words(k).capitalize()) for k in keys],
            validators=[Optional()],
        ),
    )
    setattr(
        Form,
        f"typed_value_{model.name}",
        StringField(
            "Threshold",
            description="Only this test changes. The table below shows the false alarm rate it means.",
            validators=[Optional(), number_between(1e-9, 1e9, "The threshold")],
        ),
    )
    setattr(
        Form,
        f"untype_{model.name}",
        SelectMultipleField(
            "Reset to calibrated", choices=[(k, k) for k in keys], validators=[Optional()]
        ),
    )
