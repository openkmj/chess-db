import time
from pyspark.sql import SparkSession
from dotenv import load_dotenv
import os
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    ArrayType,
)
from pyspark.sql.functions import udf, col, explode, count, mean, sum
import chess.pgn
import io

load_dotenv()

project_id = "chess-db-433217"
dataset_id = "chess_db_dataset"

spark = (
    SparkSession.builder.appName("chess")
    # .config(
    #     "spark.jars.packages",
    #     "org.apache.hadoop:hadoop-aws:3.2.4",
    # )
    # .config("spark.jars", "./redshift-jdbc42-2.1.0.30.jar")
    # .config(
    #     "spark.hadoop.fs.s3a.aws.credentials.provider",
    #     "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    # )
    # .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    # .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY"))
    # .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_KEY"))
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)

schema = StructType(
    [
        StructField("uuid", StringType(), False),
        StructField("pgn", StringType(), False),
    ]
)
game_schema = StructType(
    [
        StructField("uuid", StringType(), False),
        StructField("pgn", StringType(), False),
        StructField("white", StringType(), False),
        StructField("black", StringType(), False),
        StructField("result", IntegerType(), False),
        StructField("rating", IntegerType(), False),
    ]
)
move_schema = ArrayType(
    StructType(
        [
            StructField("fen", StringType(), False),
            StructField("position", StringType(), False),
        ]
    )
)


def parse_game(pgn, uuid):
    game = chess.pgn.read_game(io.StringIO(pgn))

    return {
        "uuid": uuid,
        "pgn": pgn,
        "white": game.headers["White"],
        "black": game.headers["Black"],
        "rating": (int(game.headers["WhiteElo"]) + int(game.headers["BlackElo"])) // 2,
        "result": (
            1
            if game.headers["Result"] == "1-0"
            else -1 if game.headers["Result"] == "0-1" else 0
        ),
    }


def parse_pgn(pgn):
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    fen = board.fen()
    position = " ".join(fen.split(" ")[:-2])
    moves = [
        {
            "fen": fen,
            "position": position,
        }
    ]
    for move in game.mainline_moves():
        board.push(move)
        fen = board.fen()
        position = " ".join(fen.split(" ")[:-2])
        moves.append(
            {
                "fen": fen,
                "position": position,
            }
        )
    return moves


# game_parquet_path = "s3a://chessdb-lake"
# redshift_url = "jdbc:redshift://default-workgroup.065512952094.ap-northeast-2.redshift-serverless.amazonaws.com:5439/dev"
# properties = {
#     "user": os.getenv("REDSHIFT_USER"),
#     "password": os.getenv("REDSHIFT_PASSWORD"),
#     "driver": "com.amazon.redshift.jdbc42.Driver",
# }


def main():
    year = "2024"
    months = ["01"]
    for month in months:
        st = time.time()
        game_df = spark.read.csv(
            f"data/games_{year}_{month}.csv",
            schema=schema,
            multiLine=True,
            quote='"',
            escape='"',
        )
        print("game_df")
        print(game_df.count())
        game_df.show(5)
        print(f"load time: {time.time() - st}")
        st = time.time()

        game_df = game_df.dropDuplicates(["uuid"])

        parse_game_udf = udf(parse_game, game_schema)

        game_df = game_df.withColumn("game", parse_game_udf(game_df.pgn, game_df.uuid))
        game_df = game_df.select("game.*")

        print("after parse_game_udf")
        # print(game_df.count())
        # game_df.show(5)
        game_df.write.parquet(f"data/parsed_games_{year}_{month}", mode="overwrite")
        print(f"parse_game_udf time: {time.time() - st}")
        st = time.time()

        # parse_pgn_udf = udf(parse_pgn, move_schema)

        # # game_df: uuid, pgn, moves
        # game_df = game_df.withColumn("moves", parse_pgn_udf(game_df.pgn))

        # move_df = game_df.select(explode("moves").alias("move"), "result").select(
        #     "move.*", "result"
        # )

        # print("after parse_pgn_udf")
        # # print(move_df.count())
        # # move_df.show(5)
        # move_df.write.parquet(f"data/parsed_pgn_{year}_{month}", mode="overwrite")
        # print(f"parse_pgn_udf time: {time.time() - st}")
        # st = time.time()

        # statistics_df = move_df.groupBy("position").agg(
        #     count("*").alias("total"),
        #     mean("result").alias("result"),
        #     # sum("result").alias(""), # python built-in sum 함수에 유의할것!
        # )

        # statistics_df.write.parquet(f"data/statistics_{year}_{month}", mode="overwrite")
        # print("total unique FEN")
        # print(statistics_df.count())
        # print(f"statistics time: {time.time() - st}")
        # st = time.time()

    spark.stop()


if __name__ == "__main__":
    main()
