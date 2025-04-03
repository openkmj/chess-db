from chessdotcom import get_titled_players, Client

# TITLED = ["GM", "WGM", "IM", "WIM", "FM", "WFM", "NM", "WNM", "CM", "WCM"]
TITLED = ["GM", "IM", "FM", "CM", "WGM", "WIM"]
# TITLED = ["GM"]

Client.request_config["headers"]["User-Agent"] = (
    "Chess.com Crawler" "Contact me at openkmj@g.skku.edu"
)


players = []

for title in TITLED:
    response = get_titled_players(title)
    players += response.json["players"]

with open("data/titled_players.csv", "w") as f:
    for player in players:
        f.write(player + ",0,0\n")


print("get user list success", len(players))
