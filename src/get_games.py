from chessdotcom import get_player_games_by_month, Client
from pyspark.sql import SparkSession
from dotenv import load_dotenv
import asyncio
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType

load_dotenv()

Client.aio = True
Client.request_config["headers"]["User-Agent"] = (
    "Chess.com Crawler" "Contact me at openkmj@g.skku.edu"
)
Client.rate_limit_handler.retries = 20  # How many times to retry a request if it fails
Client.rate_limit_handler.tts = 5  # How long to wait before retrying a request

spark = (
    SparkSession.builder.appName("chess")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)


schema = StructType(
    [
        StructField("uuid", StringType(), False),
        StructField("pgn", StringType(), False),
    ]
)


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
            cors = [
                get_player_games_by_month(user, year, month) for user, _, _ in chunk
            ]
            response = await gather_cors(cors)
            for res in response:
                try:
                    games = []
                    monthly_games = res.json["games"]
                    for game in monthly_games:
                        # game_time = game["end_time"]
                        # if game_time > timestamp:
                        if "pgn" in game and game["rated"] is True:
                            games.append(Row(uuid=game["uuid"], pgn=game["pgn"]))
                    results.extend(games)
                except:
                    pass
        except Exception as e:
            print(f"Failed while processing chunk {count}")
    return results


def main():
    # get players
    # csv format: name, timestamp
    player_df = spark.read.csv("./titled_players.csv", header=False, inferSchema=True)
    # slice only top 100 (for testing)
    # player_df = player_df.limit(2000)
    print("get players success", player_df.count())

    # data frame to list of c0, c1
    # ex) [(name1, timestamp1), (name2, timestamp2), ...]
    players = player_df.collect()

    # Row to list
    players = [(row._c0, row._c1, row._c2) for row in players]

    # players = player_df.select("_c0").rdd.flatMap(lambda x: x).collect()

    # get games
    year = "2025"
    months = ["03", "02", "01"]
    for month in months:
        games = asyncio.run(call(players, year, month))
        print("get games data success", len(games))

        game_df = spark.createDataFrame(games)
        game_df = game_df.dropDuplicates(["uuid"])
        print(game_df.count())

        # write games to parquet
        game_df.write.parquet(f"./gm_im_game_log/{month}", mode="overwrite")

    spark.stop()


if __name__ == "__main__":
    main()
