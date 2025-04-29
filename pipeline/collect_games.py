from chessdotcom import get_player_games_by_month, Client
import asyncio
import csv
from typing import List


Client.request_config["headers"]["User-Agent"] = (
    "Chess.com Crawler" "Contact me at openkmj@g.skku.edu"
)
Client.rate_limit_handler.retries = 20  # How many times to retry a request if it fails
Client.rate_limit_handler.tts = 5  # How long to wait before retrying a request

CHUNK_SIZE = 50


async def gather_cors(cors):
    return await asyncio.gather(*cors)


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


async def fetch_games(players, year, month):
    # timestamp = int(time.time()) - 24 * 60 * 60
    count = 1
    results = []
    for chunk in chunk_list(players, CHUNK_SIZE):
        try:
            print(f"Processing chunk {count}")
            count += 1
            cors = [get_player_games_by_month(user, year, month) for user in chunk]
            response = await gather_cors(cors)
            for res in response:
                try:
                    games = []
                    monthly_games = res.json["games"]
                    for game in monthly_games:
                        # game_time = game["end_time"]
                        # if game_time > timestamp:
                        if "pgn" in game and game["rated"] is True:
                            games.append((game["uuid"], game["pgn"]))
                    results.extend(games)
                except:
                    pass
        except Exception as e:
            print(f"Failed while processing chunk {count}")
    return results


def load_players(path: str) -> List[str]:
    players = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                players.append(row[0])
    return players


def save_games(games: List, path: str):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(games)


def collect_games(
    year: str,
    month: str,
    players_path="data/titled_players.csv",
    games_path="data/games",
):
    Client.aio = True
    players = load_players(players_path)
    print(f"Loaded {len(players)} players from {players_path}")
    games = asyncio.run(fetch_games(players, year, month))
    save_games(games, f"{games_path}_{year}_{month}.csv")
    print(f"Saved {len(games)} games to {games_path}_{year}_{month}.csv")
