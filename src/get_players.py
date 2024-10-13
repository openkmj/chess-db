from chessdotcom import get_titled_players, Client
import json
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()

# TITLED = ["GM", "WGM", "IM", "WIM", "FM", "WFM", "NM", "WNM", "CM", "WCM"]
TITLED = ["GM", "IM", "FM", "CM", "WGM", "WIM"]

Client.request_config["headers"]["User-Agent"] = (
    "Chess.com Crawler" "Contact me at openkmj@g.skku.edu"
)


players = []

for title in TITLED:
    response = get_titled_players(title)
    players += response.json["players"]


# write csv file
with open("titled_players2.csv", "w") as f:
    for player in players:
        f.write(player + ",0,0\n")


# MYSQL_HOST = os.getenv("MYSQL_HOST")
# MYSQL_USER = os.getenv("MYSQL_USER")
# MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")

# conn = pymysql.connect(
#     host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database="chess"
# )
# cur = conn.cursor()

# insert all players to db
# for player in players:
#     cur.execute("INSERT INTO user (name) VALUES (%s)", (player,))
# conn.commit()
# cur.close()
# conn.close()


print("get user list success", len(players))

# SQL
# create table user (
#     id int primary key auto_increment,
#     name varchar(255) not null unique
# );
