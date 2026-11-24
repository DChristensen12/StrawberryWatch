"""The Workbench's formatting helpers."""

from datetime import date, datetime

from strawberrywatch.workbench import text


def test_numbers_and_counts():
    assert text.number(8064) == "8,064"
    assert text.count(1, "site") == "1 site"
    assert text.count(4, "site") == "4 sites"
    assert text.count(1, "day") == "1 day"
    assert text.count(1, "reading") == "1 reading"
    assert text.count(8064, "reading") == "8,064 readings"
    assert text.count(1, "stretch") == "1 stretch"
    assert text.count(3, "stretch") == "3 stretches"
    assert text.count(0, "stretch") == "0 stretches"


def test_lists_have_no_oxford_comma():
    assert text.join_list(["A"]) == "A"
    assert text.join_list(["A", "B"]) == "A and B"
    assert text.join_list(["A", "B", "C"]) == "A, B and C"
    assert text.join_list([]) == ""


def test_site_groups():
    assert text.site_group(["North Fork 0"]) == "North Fork 0"
    assert text.site_group(["North Fork 0", "Oxford"]) == "North Fork 0 and 1 other site"
    assert text.site_group(["North Fork 0", "A", "B", "C"]) == "North Fork 0 and 3 other sites"


def test_long_dates():
    assert text.long_dates(date(2026, 3, 20)) == "20 March 2026"
    assert text.long_dates(date(2025, 6, 8), date(2025, 6, 14)) == "8 to 14 June 2025"
    assert text.long_dates(date(2026, 3, 31), date(2026, 4, 2)) == "31 March to 2 April 2026"
    assert (
        text.long_dates(date(2025, 12, 28), date(2026, 1, 3))
        == "28 December 2025 to 3 January 2026"
    )


def test_short_dates():
    assert text.short_dates(date(2026, 3, 20)) == "20 Mar 2026"
    assert text.short_dates(date(2026, 10, 1), date(2026, 10, 21)) == "1 to 21 Oct 2026"
    assert text.short_dates(date(2026, 3, 31), date(2026, 4, 2)) == "31 Mar to 2 Apr 2026"
    assert text.short_dates(date(2025, 12, 28), date(2026, 1, 3)) == "28 Dec 2025 to 3 Jan 2026"


def test_times():
    assert text.time_of_day(datetime(2026, 1, 1, 10, 12)) == "10:12 am"
    assert text.time_of_day(datetime(2026, 1, 1, 11, 0)) == "11 am"
    assert text.time_of_day(datetime(2026, 1, 1, 0, 0)) == "12 am"
    assert text.time_of_day(datetime(2026, 1, 1, 12, 0)) == "12 pm"
    assert text.time_of_day(datetime(2026, 1, 1, 18, 30)) == "6:30 pm"
    assert text.time_of_day(datetime(2026, 1, 1, 9, 5)) == "9:05 am"


def test_ago():
    minute, hour = 60, 3600
    assert text.ago(5 * minute) == "5 min ago"
    assert text.ago(59 * minute) == "59 min ago"
    assert text.ago(60 * minute) == "1 hour ago"
    assert text.ago(23 * hour) == "23 hours ago"
    assert text.ago(24 * hour) == "1 day ago"
    assert text.ago(47 * hour) == "1 day ago"
    assert text.ago(48 * hour) == "2 days ago"


def test_bins():
    assert text.bin_label("conductivity") == "5 µS/cm"
    assert text.bin_label("depth") == "5 mm"
    assert text.bin_label("temperature") == "0.5 °C"
    assert text.bin_label("dissolved_oxygen") == "0.5 mg/L"
    assert text.bin_label("ph") == "0.1"


def test_drift_phrase():
    names = {"north_fork_0": "North Fork 0", "oxford": "Oxford", "south_fork_1": "South Fork 1"}
    pairs = [("north_fork_0", "conductivity"), ("oxford", "conductivity")]
    assert text.drift_phrase(pairs, names.get) == "conductivity at North Fork 0 and Oxford"
    pairs += [("south_fork_1", "temperature")]
    assert (
        text.drift_phrase(pairs, names.get)
        == "conductivity at North Fork 0 and Oxford and temperature at South Fork 1"
    )


def test_manual_slugs():
    assert text.slug("What recalibrating does") == "what-recalibrating-does"
    assert len(text.MANUAL_PAGES) == 11


def test_every_manual_page_is_in_the_user_manual():
    from strawberrywatch.workbench import app

    assert sorted(app.manual_pages()) == sorted(text.slug(t) for t in text.MANUAL_PAGES)
