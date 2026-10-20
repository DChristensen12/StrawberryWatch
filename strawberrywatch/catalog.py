"""
The Models page's cards: a name, two sentences, a picture and where to read
more, kept in one place instead of spread through the templates. Pictures are
the README's, shrunk to ship with the package. Whether a model can run comes
from serving.available().
"""

from __future__ import annotations

README = "https://github.com/DChristensen12/StrawberryWatch#models"

CARDS = (
    {
        "name": "dusk_crayfish",
        "title": "Dusk Crayfish",
        "about": (
            "Forecasts each site's next conductivity reading from the whole creek, and "
            "raises an alert when the real reading is far from the forecast."
        ),
        "picture": "dusk_crayfish.jpg",
        "alt": "A crayfish on the creek bed",
    },
    {
        "name": "cobble_shoal",
        "title": "Cobble Shoal",
        "about": (
            "Checks each sensor at each site separately, so an alert names the sensor. "
            "Keeps working when some sensors are offline."
        ),
        "picture": "cobble_shoal.jpg",
        "alt": "A fish over cobbles in shallow water",
    },
    {
        "name": "flame_skimmer",
        "title": "Flame Skimmer",
        "about": (
            "Works like Cobble Shoal, but forecasts a range instead of a single value, "
            "so it can show how certain it is."
        ),
        "picture": "flame_skimmer.jpg",
        "alt": "A flame skimmer dragonfly on a stem",
    },
    {
        "name": "berry_delight",
        "title": "Berry Delight",
        "about": (
            "Planned. A logistic regression model, chosen because its results are easy to explain."
        ),
        "picture": "berry_delight.jpg",
        "alt": "Blackberries ripening on the bush",
    },
    {
        "name": "water_strider",
        "title": "Water Strider",
        "about": (
            "Checks each sensor for spikes, lasting shifts and slow drift, using statistics "
            "fitted to past data. Each alert can be explained in one sentence."
        ),
        "picture": "water_strider.jpg",
        "alt": "A water strider on the surface of the creek",
    },
    {
        "name": "crimson_parker",
        "title": "Crimson Parker",
        "about": "Name reserved for a future model.",
        "picture": "crimson_parker.jpg",
        "alt": "A white spider on a purple iris",
    },
)


# One sentence per support module, from the build spec
SUPPORT = {
    "settling_pool": "Holds back readings that look like sensor faults before the model sees them.",
    "newtwork_run": "Shows whether a change moved downstream or appeared everywhere at once.",
    "trial_bed": "Suggests the likely pollutant, when the site has the sensors to tell.",
}


def cards():
    """Every card, with can_run filled in from the serving registry."""
    from strawberrywatch.serving import available

    runnable = set(available())
    return [{**card, "can_run": card["name"] in runnable, "read_more": README} for card in CARDS]
