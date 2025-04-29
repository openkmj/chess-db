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
import pandas as pd

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
            StructField("result", IntegerType(), False),
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
    result = (
        1
        if game.headers["Result"] == "1-0"
        else -1 if game.headers["Result"] == "0-1" else 0
    )
    moves = [
        {
            "fen": fen,
            "position": position,
            "result": result,
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
                "result": result,
            }
        )
    return moves


def parse_pgn_batch(iterator):
    for df in iterator:
        rows = []
        for _, row in df.iterrows():
            parsed = parse_pgn(row["pgn"])
            for move in parsed:
                rows.append(
                    {
                        "fen": move["fen"],
                        "position": move["position"],
                        "result": move["result"],
                    }
                )
        yield pd.DataFrame(rows)


def transform_pgn_to_fen(
    year: str,
    month: str,
    games_path="data/games",
    output_path="data/position_winrate",
):
    spark = (
        SparkSession.builder.appName("pgn-to-fen")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )

    print(f"Start transforming PGN to FEN...")

    st = time.time()
    game_df = spark.read.csv(
        f"{games_path}_{year}_{month}.csv",
        schema=schema,
        multiLine=True,
        quote='"',
        escape='"',
    )

    # TODO: 중복 제거 로직 비교
    game_df = game_df.dropDuplicates(["uuid"])
    print(f"Removed duplicates time: {time.time() - st}")
    st = time.time()

    # TODO: UDF 개선
    parse_pgn_udf = udf(parse_pgn, move_schema)
    game_df = game_df.withColumn("moves", parse_pgn_udf(game_df.pgn))

    # TODO: explode vs mapInPandas
    position_df = game_df.withColumn("move", explode(game_df.moves)).select(
        "move.position", "move.result"
    )

    # position_df = game_df.mapInPandas(
    #     parse_pgn_batch,
    #     schema=StructType(
    #         [
    #             StructField("fen", StringType(), False),
    #             StructField("position", StringType(), False),
    #             StructField("result", FloatType(), False),
    #         ]
    #     ),
    # )

    position_df.cache()

    print(f"Parse pgn udf time: {time.time() - st}")
    st = time.time()

    grouped_position_df = position_df.groupBy("position").agg(
        count("*").alias("total"),
        mean("result").alias("result"),
        # sum("result").alias(""), # python built-in sum 함수에 유의할것!
    )

    grouped_position_df.write.parquet(f"{output_path}/{year}/{month}", mode="overwrite")

    print(f"Total position number: {grouped_position_df.count()}")
    print(f"Aggregation time: {time.time() - st}")

    spark.stop()
