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
    FloatType,
    LongType,
    DoubleType,
)
from pyspark.sql.functions import udf, col, explode, count, mean, sum
import chess.pgn
import io

load_dotenv()

spark = (
    SparkSession.builder.appName("chess")
    .config("spark.executor.memory", "16g")
    .config("spark.driver.memory", "12g")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)

schema = ArrayType(
    StructType(
        [
            StructField("position", StringType(), False),
            StructField("total", LongType(), False),
            StructField("result", DoubleType(), False),
        ]
    )
)


def main():
    st = time.time()
    df = spark.read.parquet("./statistics/merged")
    df.printSchema()
    print(time.time() - st)
    st = time.time()
    print(f"total position: {df.count()}")
    print(time.time() - st)
    st = time.time()

    # df = df.filter(col("total") > 20)
    # print(f"over 20 position: {df.count()}")
    # df.show(10)
    # print(time.time() - st)
    # st = time.time()

    # df = df.filter(col("total") > 50)
    # print(f"over 50 position: {df.count()}")
    # df.show(10)
    # print(time.time() - st)
    # st = time.time()

    df = df.filter(col("total") > 100)
    print(f"over 100 position: {df.count()}")
    df.show(10)
    print(time.time() - st)

    df.repartition(1).write.csv("./statistics/filtered/over100.csv", mode="overwrite")

    spark.stop()


if __name__ == "__main__":
    main()
