# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "8154e2e4-97dc-452b-a2cb-ba61bcead8da",
# META       "default_lakehouse_name": "Altschool_Hackathon_Lakehouse",
# META       "default_lakehouse_workspace_id": "9884e144-cb64-4ea4-9fca-23f975a1c8d9",
# META       "known_lakehouses": [
# META         {
# META           "id": "8154e2e4-97dc-452b-a2cb-ba61bcead8da"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# ## **ALTSCHOOL AFRICA DATA HACKATON, 2025**
# 
# ## **TEAM PAIDEIA**
# ## 
# ## **PROJECT FOCUS: EDUCATION**

# MARKDOWN ********************

# Problem statement:
# 
# Project description:
# 
# Objectives:

# MARKDOWN ********************

# ### IMPORT NECESSARY MODULES

# CELL ********************

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')a

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.sql("SELECT * FROM Altschool_Hackathon_Lakehouse.gold.student_dropout_features")
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### DATA LOADING AND INITIAL INSPECTION

# CELL ********************

# Basic dataset information
print(f"Dataset Shape: {df.count()} rows, {len(df.columns)} columns")
print("\nColumn Names and Types:")
df.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Display first few rows
df.show(5, vertical=True, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Display statistics in a Databricks table
display(df.describe())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ##### Student Identification and Demographics
# 
# **student_id**: Unique identifier for each student, ranging from 1 to 1,000,000, serving as the primary key for tracking individual student records throughout the system.
# 
# **age**: Student age in years, spanning from 14 to 30 years with an average of approximately 22 years, indicating a mix of traditional college-age students and adult learners.
# 
# **gender**: Categorical variable indicating student gender, with values including "Male" and "Female" representing the gender distribution within the student population.
# 
# **location**: Geographic identifier showing student locations within Nigeria, including cities such as Abuja and Rivers, providing insight into regional educational patterns and potential infrastructure considerations.
# 
# **device_type**: Technology platform used by students to access educational content, ranging from mobile devices like Android to desktop systems like macOS, reflecting the diverse technological landscape of the student body.
# 
# **socio_econ_status**: Socioeconomic classification of students, with categories including "High" and "Medium," providing context for understanding potential barriers to educational access and success.
# 
# **scholarship**: Boolean indicator (not present in summary but listed in schema) that likely tracks whether a student receives financial aid or scholarship support.
# 
# ##### Digital Learning Engagement Metrics
# 
# **avg_logins**: Average number of platform logins per student, ranging from approximately 5.3 to 10.4 sessions with a mean of 7.8, indicating frequency of platform engagement and digital learning commitment.
# 
# **avg_session_min**: Average duration of learning sessions in minutes, spanning from about 18 to 57 minutes with a mean of 37.6 minutes, reflecting typical study session lengths and attention spans.
# 
# **avg_video_completion**: Proportion of educational videos completed by students, ranging from 58% to 87% with an average of 72%, serving as a key indicator of content engagement and learning persistence.
# 
# **avg_discussion**: Average number of discussion forum participations or posts per student, ranging from 2.4 to 7.7 interactions with a mean of 5.0, measuring collaborative learning and peer engagement.
# 
# **avg_connectivity_issues**: Average frequency of technical connectivity problems experienced, ranging from 1.1 to 3.5 issues with a mean of 2.3, highlighting infrastructure challenges that may impact learning accessibility.
# 
# ##### Academic Performance Indicators
# 
# **avg_submission_rate**: Proportion of assignments or assessments submitted on time, ranging from 58% to 88% with an average of 73%, indicating academic compliance and engagement levels.
# 
# **avg_score_all**: Overall academic performance score across all assessments, ranging from 57.7 to 89.2 points with a mean of 73.1, representing comprehensive academic achievement.
# 
# **score_p25**: 25th percentile score, ranging from 50.3 to 84.7 with an average of 66.7, showing the lower quartile of student performance distribution.
# 
# **score_p75**: 75th percentile score, ranging from 60.8 to 95.3 with an average of 79.0, indicating the upper quartile performance benchmark.
# 
# **attendance_rate**: Proportion of classes attended, ranging from 68% to 100% with an excellent average of 91%, demonstrating strong class participation patterns.
# 
# **total_classes_missed**: Absolute number of classes not attended, ranging from 0 to 19 missed classes with an average of 5.5, providing a concrete measure of absenteeism.
# 
# ##### Psychological and Social Factors
# 
# **motivation_avg**: Self-reported or measured motivation levels on a scale, ranging from 5.0 to 9.0 with an average of 6.8, capturing student drive and academic enthusiasm.
# 
# **stress_avg**: Average stress levels experienced by students, ranging from 4.0 to 9.0 with a mean of 6.7, indicating psychological well-being and potential barriers to academic success.
# 
# **peer_interaction_avg**: Measure of social engagement with classmates, ranging from 5.0 to 15.0 interactions with an average of 10.1, reflecting collaborative learning and social integration.
# 
# **grade_trend**: Directional indicator of academic performance over time, ranging from -11.53 to +10.53 with a mean near zero (-0.001), showing whether grades are improving, declining, or stable.
# 
# ##### Predictive Analytics and Risk Assessment
# 
# **engagement_score**: Composite metric measuring overall student engagement, ranging from 5.88 to 9.15 with an average of 7.44, likely combining multiple behavioral and performance indicators.
# 
# **consistency_score**: Measure of regularity in learning behaviors and performance, ranging from 3.48 to 7.84 with a mean of 4.75, indicating reliability and predictability of student actions.
# 
# **dropout_probability**: Calculated risk score predicting likelihood of student withdrawal, ranging from 34.5% to 90.0% with an average of 66.4%, serving as an early warning system for intervention needs.
# 
# **dropout_label**: Binary classification (0 or 1) indicating actual or predicted dropout status, with nearly 100% of students labeled as 1, suggesting this dataset focuses specifically on at-risk student populations requiring targeted support interventions.


# MARKDOWN ********************

# ### DATA QUALITY ASSESSMENT

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Check for duplicate records
duplicate_count = df.count() - df.dropDuplicates().count()
print(f"\nDuplicate Records: {duplicate_count}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Check for duplicate student IDs
duplicate_student_ids = df.groupBy("student_id").count().filter(col("count") > 1).count()
print(f"Duplicate Student IDs: {duplicate_student_ids}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Data type validation
numeric_cols = ['age', 'avg_logins', 'avg_session_min', 'avg_video_completion', 
                'avg_discussion', 'avg_connectivity_issues', 'avg_submission_rate',
                'avg_score_all', 'score_p25', 'score_p75', 'attendance_rate',
                'total_classes_missed', 'motivation_avg', 'stress_avg', 
                'peer_interaction_avg', 'engagement_score', 'consistency_score',
                'dropout_probability']

categorical_cols = ['gender', 'location', 'device_type', 'socio_econ_status', 
                   'scholarship', 'grade_trend', 'dropout_label']

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Data type validation
print("\nData Type Validation:")
numeric_cols = ['age', 'avg_logins', 'avg_session_min', 'avg_video_completion', 
                'avg_discussion', 'avg_connectivity_issues', 'avg_submission_rate',
                'avg_score_all', 'score_p25', 'score_p75', 'attendance_rate',
                'total_classes_missed', 'motivation_avg', 'stress_avg', 
                'peer_interaction_avg', 'engagement_score', 'consistency_score',
                'dropout_probability']

categorical_cols = ['gender', 'location', 'device_type', 'socio_econ_status', 
                   'scholarship', 'grade_trend', 'dropout_label']

for col_name in numeric_cols:
    if col_name in df.columns:
        non_numeric = df.select(col_name).filter(~col(col_name).rlike("^-?[0-9]+\.?[0-9]*$")).count()
        print(f"{col_name}: {non_numeric} non-numeric values")

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

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
