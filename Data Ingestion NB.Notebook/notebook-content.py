# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "8154e2e4-97dc-452b-a2cb-ba61bcead8da",
# META       "default_lakehouse_name": "EduTech_DropOut_Data_Generation",
# META       "default_lakehouse_workspace_id": "9884e144-cb64-4ea4-9fca-23f975a1c8d9",
# META       "known_lakehouses": [
# META         {
# META           "id": "8154e2e4-97dc-452b-a2cb-ba61bcead8da"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# Microsoft Fabric PySpark Notebook: Synthetic Data Generator for Dropout Prediction (Edutech)
# -------------------------------------------------------------------------------------------
# Purpose: Generate large-scale, realistic synthetic datasets (millions+ rows) into Lakehouse
#          using medallion layers (bronze → silver → gold) for DA/DS consumption.
# Engine: PySpark (Fabric Notebooks)
# Author: Olayemi -- Data Engineer
# -------------------------------------------------------------------------------------------

# ============================
# 0) Imports & Spark Settings
# ============================
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

spark.conf.set("spark.sql.shuffle.partitions", "800")  # tune for your capacity
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

# For reproducibility
RANDOM_SEED = 42

# ============================
# 1) Parameters (scale here)
# ============================
# NOTE: Start with smaller numbers to validate, then scale up.
N_STUDENTS = 1_000_000   # ← scale up as needed (e.g., 3_000_000)
WEEKS = 12               # 3 months ≈ 12 weeks
SUBJECTS = ["introduction to python", "Execl", "ML", "UiUx Framework","Introduction to Java"]

REGIONS = [
    "Lagos", "Abuja", "Rivers", "Kano", "Oyo", "Kaduna", "Anambra", "Enugu",
    "Ogun", "Edo", "Delta", "Akwa Ibom", "Kwara", "Osun", "Imo", "Borno"
]
DEVICE_TYPES = ["Android", "iOS", "Windows", "macOS", "Feature Phone", "Shared Device"]
SOCIO_ECON = ["Low", "Medium", "High"]
GENDERS = ["Male", "Female"]

# Schemas / databases
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {BRONZE_SCHEMA}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER_SCHEMA}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_SCHEMA}")

# ============================
# 2) Helper Generators (column funcs)
# ============================
# NOTE: In PySpark, seed rand at call sites: F.rand(seed=...)

def categorical_from_rand(rand_col, categories):
    """Map a uniform rand() to categories with simple equal buckets."""
    n = len(categories)
    expr = F.when(rand_col < 1.0/n, F.lit(categories[0]))
    for i in range(1, n-1):
        expr = expr.when(rand_col < (i+1)/n, F.lit(categories[i]))
    expr = expr.otherwise(F.lit(categories[-1]))
    return expr

# Weighted version for socio-economic distribution (skewed towards Low/Medium)

def socio_from_rand(r):
    return (F.when(r < 0.55, F.lit("Low"))
             .when(r < 0.90, F.lit("Medium"))
             .otherwise(F.lit("High")))

# Scholarship probability influenced by socio-economic & region (toy prior)

def scholarship_prob(ses, region):
    return (F.when(ses == "Low", F.lit(0.35))
             .when(ses == "Medium", F.lit(0.18))
             .otherwise(F.lit(0.06))) + (F.when(region.isin("Lagos", "Abuja"), F.lit(-0.03)).otherwise(F.lit(0.0)))

# Logistic helper (column-safe)

def logistic(x_col):
    return F.lit(1.0) / (F.lit(1.0) + F.exp(-x_col))

# ============================
# 3) BRONZE: Base Tables
# ============================
# 3.1 Student Demographics
students = (
    spark.range(1, N_STUDENTS + 1)
    .withColumnRenamed("id", "student_id")
    .withColumn("r1", F.rand(seed=RANDOM_SEED))
    .withColumn("r2", F.rand(seed=RANDOM_SEED + 1))
    .withColumn("r3", F.rand(seed=RANDOM_SEED + 2))
    .withColumn("r4", F.rand(seed=RANDOM_SEED + 3))
    .withColumn("age", (F.floor(F.lit(14) + F.rand(seed=RANDOM_SEED) * F.lit(17))).cast("int"))  # 14-30
    .withColumn("gender", categorical_from_rand(F.col("r1"), GENDERS))
    .withColumn("location", categorical_from_rand(F.col("r2"), REGIONS))
    .withColumn("device_type", categorical_from_rand(F.col("r3"), DEVICE_TYPES))
    .withColumn("socio_econ_status", socio_from_rand(F.col("r4")))
)

students = (
    students
    .withColumn("scholarship_prob", scholarship_prob(F.col("socio_econ_status"), F.col("location")))
    .withColumn("scholarship", (F.rand(seed=RANDOM_SEED + 4) < F.col("scholarship_prob")).cast("boolean"))
    .drop("r1", "r2", "r3", "r4", "scholarship_prob")
)

students.write.mode("overwrite").saveAsTable(f"{BRONZE_SCHEMA}.student_demographics")

# 3.2 Weekly Activity Logs (per student x week)
weeks_df = spark.range(1, WEEKS + 1).withColumnRenamed("id", "week")

base_activity = (
    students.select("student_id")
    .crossJoin(weeks_df)
    .withColumn("r_a", F.rand(seed=RANDOM_SEED + 5))
    .withColumn("r_b", F.rand(seed=RANDOM_SEED + 6))
    .withColumn("r_c", F.rand(seed=RANDOM_SEED + 7))
)

# Latent risk driver (static per student) combining proxy hardships
risk_base = (
    students
    .withColumn(
        "risk_latent",
        0.6 * (F.col("socio_econ_status") == "Low").cast("int") +
        0.3 * (F.col("device_type").isin("Feature Phone", "Shared Device")).cast("int") +
        0.2 * (~F.col("scholarship")).cast("int") +
        0.1 * (F.col("location").isin("Borno", "Kano", "Kaduna")).cast("int")
    )
    .select("student_id", "risk_latent")
)

activity = (
    base_activity
    .join(risk_base, "student_id", "left")
    # logins per week: lower when risk_latent is high
    .withColumn(
        "logins_per_week",
        F.greatest(F.lit(0), F.round(F.lit(7) - F.lit(2)*F.col("risk_latent") + F.rand(seed=RANDOM_SEED + 8) * F.lit(4)).cast("int"))
    )
    .withColumn(
        "avg_session_duration_min",
        F.round(F.lit(20) + (F.lit(1) - F.col("risk_latent")) * F.lit(25) + F.rand(seed=RANDOM_SEED + 9) * F.lit(15), 1)
    )
    .withColumn(
        "video_completion_rate",
        F.round(
            F.least(
                F.lit(1.0),
                F.greatest(
                    F.lit(0.0),
                    F.lit(0.35) + (F.lit(1) - F.lit(0.3)*F.col("risk_latent")) * F.lit(0.45) + (F.rand(seed=RANDOM_SEED + 10) - F.lit(0.5)) * F.lit(0.2)
                )
            ),
            3
        )
    )
    .withColumn(
        "discussion_participation",
        F.greatest(F.lit(0), F.round((F.lit(1) - F.lit(0.5)*F.col("risk_latent")) * F.lit(5) + F.rand(seed=RANDOM_SEED + 11) * F.lit(3)).cast("int"))
    )
    .withColumn(
        "connectivity_issues",
        F.greatest(F.lit(0), F.round(F.lit(0.8) + F.lit(0.8)*F.col("risk_latent") + F.rand(seed=RANDOM_SEED + 12) * F.lit(2)).cast("int"))
    )
)

activity.write.mode("overwrite").partitionBy("week").saveAsTable(f"{BRONZE_SCHEMA}.student_activity_logs")

# 3.3 Academic Performance (per student x subject x week)
subjects_df = spark.createDataFrame([(s,) for s in SUBJECTS], ["subject"])
perf_base = students.select("student_id").crossJoin(subjects_df).crossJoin(weeks_df)

performance = (
    perf_base
    .join(risk_base, "student_id")
    .withColumn(
        "assignment_submission_rate",
        F.round(
            F.least(
                F.lit(1.0),
                F.greatest(
                    F.lit(0.0),
                    F.lit(0.85) - F.lit(0.2)*F.col("risk_latent") + (F.rand(seed=RANDOM_SEED + 13) - F.lit(0.5)) * F.lit(0.2)
                )
            ), 3
        )
    )
    .withColumn(
        "avg_score",
        F.round(
            F.least(
                F.lit(100.0),
                F.greatest(
                    F.lit(0.0),
                    F.lit(65) + (F.lit(1) - F.col("risk_latent")) * F.lit(20) + (F.rand(seed=RANDOM_SEED + 14) - F.lit(0.5)) * F.lit(25)
                )
            ), 1
        )
    )
)

performance.write.mode("overwrite").partitionBy("subject", "week").saveAsTable(f"{BRONZE_SCHEMA}.student_academic_performance")

# 3.4 Attendance (per student x week)
attendance = (
    students.select("student_id").crossJoin(weeks_df)
    .join(risk_base, "student_id")
    .withColumn("classes_scheduled", F.lit(5))
    .withColumn(
        "classes_attended",
        F.greatest(
            F.lit(0),
            F.least(
                F.col("classes_scheduled"),
                F.round(F.lit(4.3) - F.lit(0.8)*F.col("risk_latent") + F.rand(seed=RANDOM_SEED + 15) * F.lit(1.5)).cast("int")
            )
        )
    )
    .withColumn("classes_missed", (F.col("classes_scheduled") - F.col("classes_attended")))
    .withColumn("attendance_rate", F.round(F.col("classes_attended")/F.col("classes_scheduled"), 3))
)

attendance.write.mode("overwrite").partitionBy("week").saveAsTable(f"{BRONZE_SCHEMA}.student_attendance")

# 3.5 Psychosocial Surveys (per student x month-ish: weeks 4, 8, 12)
survey_weeks = spark.createDataFrame([(4,), (8,), (12,)], ["week"])
surveys = (
    students.select("student_id").crossJoin(survey_weeks)
    .join(risk_base, "student_id")
    .withColumn(
        "motivation_level",
        F.greatest(F.lit(1), F.least(F.lit(10), F.round(F.lit(7) - F.lit(2)*F.col("risk_latent") + F.rand(seed=RANDOM_SEED + 16) * F.lit(2)).cast("int")))
    )
    .withColumn(
        "stress_level",
        F.greatest(F.lit(1), F.least(F.lit(10), F.round(F.lit(4) + F.lit(2)*F.col("risk_latent") + F.rand(seed=RANDOM_SEED + 17) * F.lit(3)).cast("int")))
    )
    .withColumn(
        "peer_interaction",
        F.greatest(F.lit(0), F.round((F.lit(1) - F.lit(0.4)*F.col("risk_latent")) * F.lit(10) + F.rand(seed=RANDOM_SEED + 18) * F.lit(5)).cast("int"))
    )
)

surveys.write.mode("overwrite").partitionBy("week").saveAsTable(f"{BRONZE_SCHEMA}.student_behavioral_surveys")

# ============================
# 4) SILVER: Cleaned / Conformed
# ============================
# 4.1 Activity aggregates per student
activity_agg = (
    spark.table(f"{BRONZE_SCHEMA}.student_activity_logs")
    .groupBy("student_id")
    .agg(
        F.avg("logins_per_week").alias("avg_logins"),
        F.avg("avg_session_duration_min").alias("avg_session_min"),
        F.avg("video_completion_rate").alias("avg_video_completion"),
        F.avg("discussion_participation").alias("avg_discussion"),
        F.avg("connectivity_issues").alias("avg_connectivity_issues"),
        F.stddev_pop("logins_per_week").alias("login_volatility")
    )
)
activity_agg.write.mode("overwrite").saveAsTable(f"{SILVER_SCHEMA}.activity_agg")

# 4.2 Performance aggregates per student
perf_agg = (
    spark.table(f"{BRONZE_SCHEMA}.student_academic_performance")
    .groupBy("student_id")
    .agg(
        F.avg("assignment_submission_rate").alias("avg_submission_rate"),
        F.avg("avg_score").alias("avg_score_all"),
        F.expr("percentile_approx(avg_score, 0.25)").alias("score_p25"),
        F.expr("percentile_approx(avg_score, 0.75)").alias("score_p75")
    )
)
perf_agg.write.mode("overwrite").saveAsTable(f"{SILVER_SCHEMA}.performance_agg")

# 4.3 Attendance aggregates per student
attendance_agg = (
    spark.table(f"{BRONZE_SCHEMA}.student_attendance")
    .groupBy("student_id")
    .agg(
        F.avg("attendance_rate").alias("attendance_rate"),
        F.sum("classes_missed").alias("total_classes_missed")
    )
)
attendance_agg.write.mode("overwrite").saveAsTable(f"{SILVER_SCHEMA}.attendance_agg")

# 4.4 Behavioral aggregates per student
behavior_agg = (
    spark.table(f"{BRONZE_SCHEMA}.student_behavioral_surveys")
    .groupBy("student_id")
    .agg(
        F.avg("motivation_level").alias("motivation_avg"),
        F.avg("stress_level").alias("stress_avg"),
        F.avg("peer_interaction").alias("peer_interaction_avg")
    )
)
behavior_agg.write.mode("overwrite").saveAsTable(f"{SILVER_SCHEMA}.behavior_agg")

# 4.5 Score trend (last 4 weeks - first 4 weeks) as simple slope proxy
perf_weeks = (
    spark.table(f"{BRONZE_SCHEMA}.student_academic_performance")
    .groupBy("student_id", "week")
    .agg(F.avg("avg_score").alias("week_avg_score"))
)

first4 = perf_weeks.filter(F.col("week") <= 4).groupBy("student_id").agg(F.avg("week_avg_score").alias("score_first4"))
last4  = perf_weeks.filter(F.col("week") > 8).groupBy("student_id").agg(F.avg("week_avg_score").alias("score_last4"))
trend = (
    first4.join(last4, "student_id", "outer")
    .fillna({"score_first4": 60.0, "score_last4": 60.0})
    .withColumn("grade_trend", F.round(F.col("score_last4") - F.col("score_first4"), 2))
)

trend.write.mode("overwrite").saveAsTable(f"{SILVER_SCHEMA}.grade_trend")

# ============================
# 5) GOLD: ML Feature Table + Label
# ============================
# Label generation (dropout) using logistic on risk + engagement & performance signals
silver_join = (
    students
    .join(activity_agg, "student_id", "left")
    .join(perf_agg, "student_id", "left")
    .join(attendance_agg, "student_id", "left")
    .join(behavior_agg, "student_id", "left")
    .join(trend, "student_id", "left")
)

# Build a composite risk score
risk_signal = (
    F.lit(1.2)*(F.lit(1) - (F.col("avg_video_completion"))) +
    F.lit(1.0)*(F.coalesce(F.col("login_volatility"), F.lit(0.0))) +
    F.lit(0.8)*(F.lit(1) - F.col("attendance_rate")) +
    F.lit(1.0)*(F.when(F.col("avg_score_all") < 55, F.lit(1)).otherwise(F.lit(0))) +
    F.lit(0.6)*(F.when(F.col("grade_trend") < -5, F.lit(1)).otherwise(F.lit(0))) +
    F.lit(0.5)*(F.when(F.col("stress_avg") > 6, F.lit(1)).otherwise(F.lit(0))) +
    F.lit(0.4)*(F.when(F.col("motivation_avg") < 5, F.lit(1)).otherwise(F.lit(0)))
)

# Convert to probability via logistic; tune intercept to target ~15-25% dropout rate
p_dropout = logistic(F.lit(-1.2) + risk_signal)

gold = (
    silver_join
    .withColumn("engagement_score", F.round(F.lit(0.4)*F.col("avg_logins") + F.lit(0.6)*F.col("avg_video_completion")*F.lit(10), 3))
    .withColumn("consistency_score", F.round(F.lit(10)/(F.lit(1)+F.col("login_volatility")), 3))
    .withColumn("dropout_probability", p_dropout)
    .withColumn("dropout_label", (F.col("dropout_probability") > F.lit(0.35)).cast("int"))
    .select(
        "student_id", "age", "gender", "location", "device_type", "socio_econ_status", "scholarship",
        "avg_logins", "avg_session_min", "avg_video_completion", "avg_discussion", "avg_connectivity_issues",
        "avg_submission_rate", "avg_score_all", "score_p25", "score_p75",
        "attendance_rate", "total_classes_missed",
        "motivation_avg", "stress_avg", "peer_interaction_avg",
        "grade_trend",
        "engagement_score", "consistency_score",
        F.round(F.col("dropout_probability"), 4).alias("dropout_probability"),
        "dropout_label"
    )
)

# Persist GOLD features
(
    gold.write
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(f"{GOLD_SCHEMA}.student_dropout_features")
)




# I created view in the Gold layer so the DA/DS can work on it, without affecting the main Data
# Create SQL view for DA/DS convenience
spark.sql(f"""
CREATE OR REPLACE VIEW {GOLD_SCHEMA}.vw_student_dropout_features AS
SELECT * FROM {GOLD_SCHEMA}.student_dropout_features
""")

# ============================
# 6) Quality & Row Counts (sanity checks)
# ============================
counts = {
    "students": spark.table(f"{BRONZE_SCHEMA}.student_demographics").count(),
    "activity": spark.table(f"{BRONZE_SCHEMA}.student_activity_logs").count(),
    "performance": spark.table(f"{BRONZE_SCHEMA}.student_academic_performance").count(),
    "attendance": spark.table(f"{BRONZE_SCHEMA}.student_attendance").count(),
    "surveys": spark.table(f"{BRONZE_SCHEMA}.student_behavioral_surveys").count(),
    "gold_features": spark.table(f"{GOLD_SCHEMA}.student_dropout_features").count()
}

print("Row counts:", counts)

# ============================
# 7) Scale-up Notes
# ============================
# - Increase N_STUDENTS and adjust spark.sql.shuffle.partitions accordingly.
# - Consider writing with partitioning on demographic columns (e.g., location) for GOLD
#   if you plan to do region-sliced modeling/dashboards.
# - Use OPTIMIZE / ZORDER (if available) on large Delta tables to speed up queries.
# - For streaming-like demos, generate week-by-week and append to BRONZE activity.

# Example (append next batch):
# next_activity_batch.write.mode("append").partitionBy("week").saveAsTable(f"{BRONZE_SCHEMA}.student_activity_logs")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================
# Update Subjects to EdTech Context
# ============================

# Define new subject list
new_subjects = [
    "Introduction to Python",
    "Excel",
    "ML",
    "UI/UX Framework",
    "Introduction to Java"
]

# Load Bronze table
academic_df = spark.table(f"{BRONZE_SCHEMA}.student_academic_performance")

# Build Spark SQL expression for subject array
subjects_sql_array = ",".join([f"'{s}'" for s in new_subjects])

# Replace subjects with random choice from new_subjects
updated_academic_df = academic_df.withColumn(
    "subject",
    F.expr(f"element_at(array({subjects_sql_array}), int(rand()*{len(new_subjects)}+1))")
)

# Overwrite Bronze table with updated subjects
(
    updated_academic_df.write
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(f"{BRONZE_SCHEMA}.student_academic_performance")
)

print("✅ Subjects updated successfully to EdTech courses!")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
