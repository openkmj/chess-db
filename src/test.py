from pyspark.sql import SparkSession
import random

spark = (
    SparkSession.builder.appName("Test").config("spark.driver.bindAddress", "0.0.0.0")
    # .config("spark.dynamicAllocation.enabled", "false")
    .getOrCreate()
)


df = spark.createDataFrame(
    [(random.randint(0, 100), random.randint(0, 100)) for x in range(1000)],
    ["x", "y"],
)
df.show(10)

spark.stop()
