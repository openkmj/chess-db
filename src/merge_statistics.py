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
    months = ["08", "07", "06", "05", "04", "03", "02", "01"]
    # months = ["08", "07"]
    files = [f"./statistics/{month}" for month in months]

    st = time.time()
    df = spark.read.parquet(*files)
    df.printSchema()
    print(time.time() - st)
    st = time.time()

    df = df.groupBy("position").agg(
        sum("total").alias("total_sum"),
        sum(col("result") * col("total")).alias("weighted_sum"),
    )
    df = (
        df.withColumn("result", col("weighted_sum") / col("total_sum"))
        .drop("weighted_sum")
        .withColumnRenamed("total_sum", "total")
    )

    df.write.parquet("./statistics/merged", mode="overwrite")
    print(time.time() - st)

    # merge all parquet data
    # ./statistics/08
    # ./statistics/07
    # ...
    # ./statistics/01

    # after load, group by

    spark.stop()


if __name__ == "__main__":
    main()
