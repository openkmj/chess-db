from chessdotcom import get_titled_players, Client
from typing import List
from enum import Enum


class Title(str, Enum):
    GM = "GM"
    WGM = "WGM"
    IM = "IM"
    WIM = "WIM"
    FM = "FM"
    WFM = "WFM"
    NM = "NM"
    WNM = "WNM"
    CM = "CM"
    WCM = "WCM"


DEFAULT_TITLE_LIST = [Title.GM]


Client.request_config["headers"]["User-Agent"] = (
    "Chess.com Crawler" "Contact me at openkmj@g.skku.edu"
)


def collect_titled_players(
    titles: List[Title] = DEFAULT_TITLE_LIST, output_path="data/titled_players.csv"
):
    players = []

    for title in titles:
        response = get_titled_players(title.value)
        players += response.json["players"]

    with open(output_path, "w") as f:
        for player in players:
            f.write(player + ",0,0\n")

    print(f"Player list successfully collected: {len(players)} players → {output_path}")
