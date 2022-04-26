import pytest


@pytest.fixture(scope="session")
def label_dictionaries():
    return {
        "label_a": [
            "4wd",
            "beach",
            "country",
            "nature",
            "offroad",
            "winter",
            "terrain",
            "disconnect",
            "adventure",
            "vehicle",
        ],
        "label_b": ["cost", "pricey", "price", "consumer", "reverse", "value"],
    }


@pytest.fixture(scope="session")
def wrong_label_dictionaries():
    return {
        "label_a": ["kitchen", "bathroom"],
        "label_b": ["ocean", "moon"],
    }


@pytest.fixture(scope="session")
def text():
    return """Jeep's product range consists solely of sport utility vehicles – both crossovers and fully off-road worthy SUVs and models, including one pickup truck. Previously, Jeep's range included other pick-ups, as well as small vans, and a few roadsters. Some of Jeep's vehicles—such as the Grand Cherokee—reach into the luxury SUV segment, a market segment the 1963 Wagoneer is considered to have started.[5] Jeep sold 1.4 million SUVs globally in 2016, up from 500,000 in 2008,[6][7] two-thirds of which in North America,[8] and was Fiat-Chrysler's best selling brand in the U.S. during the first half of 2017.[9] In the U.S. alone, over 2400 dealerships hold franchise rights to sell Jeep-branded vehicles, and if Jeep were spun off into a separate company, it is estimated to be worth between $22 and $33.5 billion—slightly more than all of FCA (US).[8][7]"""
