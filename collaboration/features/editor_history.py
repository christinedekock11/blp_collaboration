import pandas as pd
import numpy as np

from ..utils import entropy

from pyspark.sql import functions as f
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, IntegerType, DoubleType, FloatType

def talk_article_ratio(page_namespace_list):
    page_namespace_list = np.array(page_namespace_list)
    talk = np.sum(page_namespace_list == 1) + 1
    article = np.sum(page_namespace_list == 0) + 1
    return float(article / talk)


udf_page_talk_ratio = udf(talk_article_ratio, FloatType())


def contribution_fracs(page_ids):
    counts = np.unique(page_ids, return_counts=True)[1]
    cf = counts / sum(counts)
    entropy_cf = float(entropy(cf))
    return entropy_cf


udf_contribution_frac = udf(contribution_fracs, FloatType())


def read_revisions(filename, rename=False):
    revisions = pd.read_json(filename, lines=True)
    if rename:
        revisions = revisions.rename(columns={'revision_timestamp': 'event_timestamp', 'user_id': 'event_user_id'})
    revisions['event_timestamp'] = pd.to_datetime(revisions['event_timestamp'])
    revisions = revisions.sort_values(by='event_timestamp', ascending=True)
    return revisions


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def get_editor_features(tag_user_histories):
    tag_features = tag_user_histories.groupby('event_user_id') \
        .agg(f.last('num_groups').alias('num_groups'),
             f.countDistinct('page_id').alias('num_articles'),
             f.count('revision_id').alias('num_edits'),
             f.last('num_blocks_historical').alias('num_past_blocks'),
             f.last('num_curr_blocks').alias('num_curr_blocks'),
             f.sum(col("is_revert_bool")).alias('num_reverts_by_others'),
             f.sum(col('is_reverted_bool')).alias('num_reverts_of_others'),
             f.last('days_since_registration').alias('time_since_registration'),
             udf_page_talk_ratio(f.collect_list('page_namespace')).alias('talk_article_ratio'),
             udf_contribution_frac(f.collect_list('page_id')).alias('contribution_frac_entropy')
             )

    return tag_features


def get_user_interactions(user_histories_1year):
    user_page_revisions = user_histories_1year.select(col('page_id'), col('event_user_id'), col('revision_id'),
                                                      col('page_namespace')) \
        .groupBy("page_id", "event_user_id").agg(
        f.count("revision_id").alias("revisions_count"),
        f.first("page_namespace").alias('page_namespace'))

    self_join_df = user_page_revisions.toDF(*[c + '_r' for c in user_page_revisions.columns])
    editor_interactions = user_page_revisions.join(self_join_df, [user_page_revisions.page_id == self_join_df.page_id_r,
                                                                  user_page_revisions.event_user_id != self_join_df.event_user_id_r]).drop(
        'page_id_r')

    return editor_interactions


def calculate_concentration_ratios(editor_interactions):
    concentration_ratio = editor_interactions.groupby('page_id').agg(
        f.countDistinct('event_user_id').alias('num_editors'),
        f.sum('revisions_count').alias('num_revisions')) \
        .withColumn('concentration_ratio', col('num_editors') / col('num_revisions'))

    editor_interactions = editor_interactions.join(concentration_ratio, on='page_id')
    return editor_interactions


def get_directed_features(paired_interactions):
    user_article_edits = paired_interactions.groupby('event_user_id') \
        .agg(f.sum('num_common_articles').alias('editor_pages_total'),
             f.sum('num_revisions_articles').alias('editor_revisions_total'))

    directed = paired_interactions \
        .join(user_article_edits.select('event_user_id', 'editor_pages_total', 'editor_revisions_total'),
              on="event_user_id") \
        .withColumn("coedit_ratio", (col("num_common_articles") / col("editor_pages_total"))) \
        .withColumn('coedit_revisions_ratio', (col('num_revisions_articles') / col('editor_revisions_total'))) \
        .select('event_user_id', 'event_user_id_r', 'coedit_ratio', 'coedit_revisions_ratio')

    return directed


def get_undirected_features(paired_interactions, paired_interactions_articles):
    features_all = paired_interactions.withColumn('pair',
                                                  f.array_sort(f.array(col('event_user_id'), col('event_user_id_r')))) \
        .drop_duplicates(subset=['pair']).select('pair', 'num_common_pages')
    features_articles = paired_interactions_articles.withColumn('pair', f.array_sort(
        f.array(col('event_user_id'), col('event_user_id_r')))) \
        .drop_duplicates(subset=['pair']) \
        .select('pair', 'num_common_articles')  # ,'mean_concentration_ratio')

    undirected = features_all.join(features_articles, on='pair')
    return undirected


def calculate_collaboration_features(editor_interactions):
    # editor_interactions = calculate_concentration_ratios(editor_interactions)

    paired_interactions_articles = editor_interactions.filter(col('page_namespace') == 0) \
        .groupby('event_user_id', 'event_user_id_r') \
        .agg(f.count("page_id").alias('num_common_articles'),
             f.sum('revisions_count').alias('num_revisions_articles')) \
        .cache()
    # f.mean('concentration_ratio').alias('mean_concentration_ratio'))\
    # .filter(col('num_common_articles')>=5)

    paired_interactions_all = editor_interactions.groupby('event_user_id', 'event_user_id_r') \
        .agg(f.count("page_id").alias('num_common_pages'))

    directed_features = get_directed_features(paired_interactions_articles)
    undirected_features = get_undirected_features(paired_interactions_all, paired_interactions_articles)

    return directed_features, undirected_features
