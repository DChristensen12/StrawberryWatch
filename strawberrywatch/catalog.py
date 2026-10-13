"""
The Models page's cards: a name, two plain sentences, a picture and where to read
more. One place to fix the words, rather than spread through templates.

Pictures are the README's, under strawberrywatch/dashboard/static/models/,
shrunk to ship with the package. Whether a model can run isn't written here:
it's whatever serving.available() says, so a model appears as runnable once
it's registered and passes the conformance tests.
"""

from __future__ import annotations

README = "https://github.com/DChristensen12/StrawberryWatch#models"

CARDS = (
    {
        "name": "dusk_crayfish",
        "title": "Dusk Crayfish",
        "about": (
            "Learns how the creek normally behaves as a whole and notices when readings "
            "stop matching. It predicts each site's next reading and watches how far "
            "the real one lands from it."
        ),
        "picture": "dusk_crayfish.jpg",
        "alt": "A crayfish on the creek bed",
    },
    {
        "name": "cobble_shoal",
        "title": "Cobble Shoal",
        "about": (
            "Checks every sensor at every site on its own, so it can say which probe "
            "looks wrong. It keeps working when some sensors go quiet."
        ),
        "picture": "cobble_shoal.jpg",
        "alt": "A fish over cobbles in shallow water",
    },
    {
        "name": "flame_skimmer",
        "title": "Flame Skimmer",
        "about": (
            "A planned model that says how sure it is about every prediction. It would "
            "learn a range for each forecast instead of a single number."
        ),
        "picture": "flame_skimmer.jpg",
        "alt": "A flame skimmer dragonfly on a stem",
    },
    {
        "name": "berry_delight",
        "title": "Berry Delight",
        "about": (
            "A planned model built on logistic regression, one of the simplest methods "
            "there is. Its answers would be easy to explain."
        ),
        "picture": "berry_delight.jpg",
        "alt": "Blackberries ripening on the bush",
    },
    {
        "name": "water_strider",
        "title": "Water Strider",
        "about": "A name saved for a future model. What kind hasn't been decided yet.",
        "picture": "water_strider.jpg",
        "alt": "A water strider on the surface of the creek",
    },
    {
        "name": "crimson_parker",
        "title": "Crimson Parker",
        "about": "Another name saved for a future model. What kind hasn't been decided yet.",
        "picture": "crimson_parker.jpg",
        "alt": "A white spider on a purple iris",
    },
)


# One plain sentence per support module, from the build spec's own descriptions
SUPPORT = {
    "settling_pool": "Flags readings not to trust and holds them back before the model reads them.",
    "newtwork_run": "Says whether something travelled downstream or showed up everywhere at once.",
    "trial_bed": "Names the pollutant the readings look most like, when the sensors can tell.",
}


def cards():
    """Every card, with can_run filled in from the serving registry."""
    from strawberrywatch.serving import available

    runnable = set(available())
    return [{**card, "can_run": card["name"] in runnable, "read_more": README} for card in CARDS]
