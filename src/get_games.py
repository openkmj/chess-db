from chessdotcom import get_player_games_by_month, Client
import asyncio
import csv

Client.aio = True
Client.request_config["headers"]["User-Agent"] = (
    "Chess.com Crawler" "Contact me at openkmj@g.skku.edu"
)
Client.rate_limit_handler.retries = 20  # How many times to retry a request if it fails
Client.rate_limit_handler.tts = 5  # How long to wait before retrying a request


async def gather_cors(cors):
    return await asyncio.gather(*cors)


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


async def call(players, year, month):
    # timestamp = int(time.time()) - 24 * 60 * 60
    count = 1
    results = []
    for chunk in chunk_list(players, 50):
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


def load_players():
    players = []
    with open("data/titled_players.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                players.append(row[0])
    return players


def save_games(games, path):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(games)


def main():
    players = load_players()
    print("get user list success", len(players))

    year = "2023"
    months = ["12", "11", "10", "09", "08", "07", "06", "05", "04", "03", "02", "01"]
    for month in months:
        games = asyncio.run(call(players, year, month))
        print(f"get games data success {year}-{month}", len(games))
        save_games(games, f"data/games_{year}_{month}.csv")


if __name__ == "__main__":
    main()
