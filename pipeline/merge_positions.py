import time
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType,
    LongType,
    DoubleType,
)
from pyspark.sql.functions import udf, col, explode, count, mean, sum

schema = ArrayType(
    StructType(
        [
            StructField("position", StringType(), False),
            StructField("total", LongType(), False),
            StructField("result", DoubleType(), False),
        ]
    )
)


def merge_positions():
    spark = (
        SparkSession.builder.appName("chess")
        .config("spark.executor.memory", "16g")
        .config("spark.driver.memory", "16g")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )

    print("Starting merge positions")
    st = time.time()

    months = ["08", "07", "06", "05", "04", "03", "02", "01"]
    # months = ["08", "07"]
    files = [f"./statistics/{month}" for month in months]

    df = spark.read.parquet(*files)
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

    spark.stop()
