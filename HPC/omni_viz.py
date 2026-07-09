import os
import glob

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from itertools import chain

DATA_ROOT = "/mnt/parscratch/users/acp25tw/datasets/OmniMedVQA-V2"
CPUS = os.environ.get("SLURM_CPUS_PER_TASK", "8")

spark = (
    SparkSession.builder
    .appName("OmniMedVQA-V2")
    .master(f"local[{CPUS}]")
    .config("spark.local.dir","/mnt/parscratch/users/acp25tw")
    .getOrCreate()
)



print("Spark version:", spark.version)
print("Using CPUs:", CPUS)

print("Reading parquet files from:", DATA_ROOT)
parquet_files = sorted(glob.glob(os.path.join(DATA_ROOT, "*", "*.parquet")))
print("Number of parquet files:", len(parquet_files))

print("Loading data from parquet files...")
df = spark.read.parquet(*parquet_files)
print("Loaded data")
print("\nSchema:")
df.printSchema()
print("\nTotal rows:")
print(df.count())

df = (
    df.withColumn("_source_file", F.input_file_name())
    .withColumn(
        "subset",
        F.regexp_extract(F.col("_source_file"), r"/OmniMedVQA-V2/([^/]+)/", 1)
    )
    .withColumn(
        "split",
        F.regexp_extract(F.col("_source_file"), r"/(train|test)-", 1)
    )
    .withColumn(
        "category",
        F.when(F.col("subset").startswith("mod-"), F.lit("modality"))
         .when(F.col("subset").startswith("qt-"), F.lit("question_type"))
         .otherwise(F.lit("unknown"))
    )
)

print("Added subset, split, and category columns")

df.groupBy("subset", "category", "split") \
  .count() \
  .orderBy("category", "subset", "split") \
  .show(100, truncate=False)

# Clean data : modality column
df = df.withColumn("modality_raw", F.col("modality"))

modality_map = {
    "MR (Mag-netic Resonance Imaging)": "MRI",
    "MR (Magnetic Resonance Imaging)": "MRI",
    "OCT (Optical Coherence Tomography": "OCT",
    "OCT (Optical Coherence Tomography)": "OCT",
    "Ultrasound": "Ultrasound",
    "ultrasound": "Ultrasound",
    "CT(Computed Tomography)": "CT",
    "Fundus Photography": "Fundus",
    "Microscopy Images": "Microscopy",
    "Dermoscopy": "Dermoscopy",
    "X-Ray": "X-Ray",
}

mapping_expr = F.create_map(
    [F.lit(x) for x in chain(*modality_map.items())]
)

df = (
    df.withColumn("modality_raw", F.col("modality"))
      .withColumn(
          "modality",
          F.coalesce(mapping_expr[F.col("modality_raw")], F.col("modality_raw"))
      )
)
df = df.drop("modality_raw")
modality_counts = (
    df.groupBy("modality")
      .count()
      .orderBy("modality")
)

n_modalities = modality_counts.count()
print("Number of distinct normalised modality values:", n_modalities)
modality_counts.show(n_modalities, truncate=False)

# Create image hash (since image path is unavailable)
df = df.withColumn(
    "image_hash",
    F.sha2(F.col("image.bytes"), 256)
)

# Separate two category
df_mod = df.filter(F.col("category") == "modality")
df_qt = df.filter(F.col("category") == "question_type")

df_mod_len = df_mod.count()
df_qt_len = df_qt.count()

print("df rows:", df.count())
print("df_mod rows:", df_mod_len)
print("df_qt rows:", df_qt_len)
print("df_mod + df_qt:", df_mod_len + df_qt_len)

df_mod.groupBy("modality").count().orderBy("modality").show(truncate=False)

# Find extra sample
key_cols = [
    "image_hash",
    "problem",
    "answer_letter",
]

mod_keys = df_mod.select(key_cols)
qt_keys = df_qt.select(key_cols)

qt_not_in_mod = qt_keys.exceptAll(mod_keys)
mod_not_in_qt = mod_keys.exceptAll(qt_keys)

print("Rows in df_qt but not df_mod:", qt_not_in_mod.count())
print("Rows in df_mod but not df_qt:", mod_not_in_qt.count())



print("\nExtra row founds as :")
cols_to_show = [c for c in qt_not_in_mod.columns if c not in ["image", "row_key"]]
qt_not_in_mod.select(cols_to_show).show(qt_not_in_mod.count(), truncate=80, vertical=True)

# Data distribution
def show_dist(dataframe, column_name, sort_by_count=True):
    total = dataframe.count()

    result = (
        dataframe
        .groupBy(column_name)
        .count()
        .withColumn("percent", F.round(F.col("count") / F.lit(total) * 100, 2))
    )

    if sort_by_count:
        result = result.orderBy(F.desc("count"))
    else:
        result = result.orderBy(column_name)

    result.show(truncate=False)
    return result

print("Category : Modality")
print("Modality Distribution:")
modality_dist = show_dist(df_mod, "modality")
print("Question Type Distribution:")
question_type_dist = show_dist(df_mod, "question_type")
print("Answer Distribution (Modality):")
answer_dist_mod = show_dist(df_mod, "answer_letter", sort_by_count=False)
print("Category : Question Type")
print("Modality Distribution:")
modality_dist_qt = show_dist(df_qt, "modality")
print("Question Type Distribution:")
question_type_dist_qt = show_dist(df_qt, "question_type")
print("Answer Distribution (Question Type):")
answer_dist_qt = show_dist(df_qt, "answer_letter", sort_by_count=False)

spark.stop()