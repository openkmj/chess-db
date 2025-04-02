TARGET_FILE = ["./statistics/03", "./statistics/02", "./statistics/01"]
OUTPUT_FILE = "./statistics.csv"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

spark = (
    SparkSession.builder.appName("chess")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)


def main():
    # df = spark.read.parquet("./statistics")
    df = spark.read.parquet(*TARGET_FILE)

    # result * total을 먼저 계산해서 나중에 weighted sum 계산에 사용
    df = df.withColumn("weighted_result", col("result") * col("total"))

    aggregated_df = (
        df.groupBy("position")
        .agg(
            spark_sum("weighted_result").alias("sum_weighted_result"),
            spark_sum("total").alias("total_sum"),
        )
        .withColumn("result", col("sum_weighted_result") / col("total_sum"))
        .select("position", "result", "total_sum")
        .withColumnRenamed("total_sum", "total")
    )

    # aggregated_df.write.csv(OUTPUT_FILE, header=True)
    aggregated_df.write.parquet("./final", mode="overwrite")
    print("success")


# main()


def main2():
    df = spark.read.parquet("./final")
    print(df.count())
    df = df.filter(col("total") == 6)
    print(df.count())
    df.coalesce(1).write.csv("./test_data", header=True)
    df.show(5)


# def main3():
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     df = pd.read_csv("./train_data.csv")
#     print("FEN 예시:", df["position"].iloc[0])
#     print("Winrate 분포:")
#     print(df["result"].describe())

#     # 히스토그램
#     df["result"].hist(bins=20)
#     plt.title("Winrate Histogram")
#     plt.show()


main2()
