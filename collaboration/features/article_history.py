import pandas as pd
pd.options.mode.chained_assignment = None
import datetime
import numpy as np
import matplotlib.pylab as plt
from math import log2

# group by page, namespace = 0
TIMEFRAME_DAYS = 7

def article_age_years(page_revisions):
    diff = page_revisions.event_timestamp.max() - page_revisions.event_timestamp.min()
    diff = diff // datetime.timedelta(days=365.2425)
    return diff


def curr_size(page_revisions):
    return page_revisions.iloc[-1].revision_text_bytes


def frac_minor_edits(page_revisions):
    return page_revisions.revision_minor_edit.mean()


def last_edit_size(page_revisions):
    #edits = page_revisions[-2:].revision_text_bytes.values
    #return edits[0] / edits[1]
    return page_revisions.revision_text_bytes.values[-1]


def num_revisions(page_revisions):
    return len(page_revisions)


def num_editors(page_revisions):
    return page_revisions.event_user_id.nunique()


def frac_recent_revisions(page_revisions):
    start_time = page_revisions.event_timestamp.max() - datetime.timedelta(days=2)
    recent = page_revisions[page_revisions.event_timestamp > start_time]
    return len(recent) / len(page_revisions)


def top_contributor_frac(page_revisions):
    counts = page_revisions.event_user_id.value_counts().dropna().sort_values(ascending=False)
    if len(counts) > 0:
        cf = counts.iloc[0] / sum(counts)
    else:
        cf = 0
    return cf


def contribution_frac_entropy(page_revisions):
    counts = page_revisions.event_user_id.value_counts().dropna()
    cf = list(counts / sum(counts))
    cf_ent = entropy(cf)
    return cf_ent

def frac_anon_revisions(page_revisions):
    return page_revisions.event_user_id.isna().mean()


def concentration_ratio(page_revisions):
    return page_revisions.event_user_id.nunique() / len(page_revisions)


def entropy(p):
    return -sum([p[i] * log2(p[i]) for i in range(len(p))])

def calculate_page_metrics(revisions):
    revisions['time_to_respond'] = revisions['event_timestamp'].diff().dt.total_seconds().abs().div(60 * 60,
                                                                                                    fill_value=0).round()
    revisions['time_responded_to'] = revisions['event_timestamp'].diff(-1).dt.total_seconds().abs().div(60 * 60,
                                                                                                        fill_value=0).round()
    revisions['edit_size'] = revisions['revision_text_bytes'].diff().fillna(revisions.revision_text_bytes).abs()
    revisions['next_edit_size'] = revisions['revision_text_bytes'].diff()[1:].abs().tolist() + [0]
    return revisions


def get_user_article_features(revisions):
    user_article = revisions.groupby('event_user_id')
    means = user_article[['edit_size', 'next_edit_size', 'time_to_respond', 'time_responded_to']].mean()
    counts = user_article.size().reset_index().rename(columns={0: 'num_edits'}).set_index('event_user_id')
    counts['frac_page_edits'] = counts.num_edits / len(revisions)
    reverts = user_article[['revision_is_identity_reverted', 'revision_is_identity_revert']].sum()
    user_article = pd.concat([means, counts, reverts], axis=1).reset_index().to_dict('records')
    return user_article


article_feature_functions = [article_age_years, curr_size, frac_minor_edits, last_edit_size, num_revisions,
                                 num_editors, frac_recent_revisions, top_contributor_frac, frac_anon_revisions,
                                 concentration_ratio, contribution_frac_entropy]
def get_talk_features(revisions):
    features = {f.__name__: f(revisions) for f in
                [num_revisions, frac_recent_revisions, num_editors, top_contributor_frac]}

    features['mean_edit_size'] = revisions.edit_size.mean()
    features['mean_response_time'] = revisions.time_to_respond.mean()

    return features


def get_article_features(revisions, tag_date, days=2):
    means = revisions[['edit_size', 'time_to_respond']].mean().to_dict()
    page_features = {f.__name__: f(revisions) for f in article_feature_functions}

    recent_revisions = revisions[revisions.event_timestamp >
                                          (tag_date - datetime.timedelta(days=days))]
    rec_means = recent_revisions[['edit_size', 'time_to_respond']].mean() \
        .rename({'edit_size': 'recent_edit_size', 'time_to_respond': 'recent_response_time'}).to_dict()

    return {**means, **page_features, **rec_means}


def daily_activity(data, start_date):
    data['days_since_start'] = (data['event_timestamp'] - start_date).dt.days
    daily_edit_count = data.groupby('days_since_start').size()
    daily_edit_vol = data.groupby('days_since_start')['edit_size'].apply(lambda x: x.abs().sum())
    df = pd.DataFrame({'daily_edit_count': daily_edit_count, 'daily_edit_vol': daily_edit_vol})
    df = df.reindex(np.arange(df.index.min(), df.index.max() + 1)).fillna(0)

    return df

def get_talk_features(tag_talk_revisions):
    tag_talk_revisions = calculate_page_metrics(tag_talk_revisions)

    features = {f.__name__: f(tag_talk_revisions) for f in
                [num_revisions, frac_recent_revisions, num_editors, top_contributor_frac]}

    features['mean_edit_size'] = tag_talk_revisions.edit_size.mean()
    features['mean_response_time'] = tag_talk_revisions.time_to_respond.mean()

    return features

def talk_vs_article_activity(talk, article, title, smoothing):
    start_date = article.event_timestamp.min()
    talk_activity = daily_activity(talk, start_date)
    article_activity = daily_activity(article, start_date)

    article_activity = article_activity[title].rolling(smoothing).mean().fillna(0)
    talk_activity = talk_activity[title].rolling(smoothing).mean().fillna(0)

    template_added = article[article['has_template'].diff() == 1].days_since_start.values
    template_removed = article[article['has_template'].diff() == -1].days_since_start.values

    return article_activity, talk_activity, template_added, template_removed


def plot_daily_activity(talk, article, title='daily_edit_vol', smoothing=60):
    article_act, talk_act, temp_add, temp_rem = talk_vs_article_activity(talk, article, title, smoothing)

    plt.figure(figsize=[12, 7])
    plt.scatter(talk_act.index.values, np.log(talk_act.values + 1), label='Talk', alpha=0.9)
    plt.plot(talk_act.index.values, np.log(talk_act.values + 1))
    plt.scatter(article_act.index.values, np.log(article_act.values + 1), label='Article', alpha=0.9)
    plt.plot(article_act.index.values, np.log(article_act.values + 1))

    for i, t in enumerate(temp_rem):
        plt.axvline(x=t, ls='--', c='b', label='template_removed' if i == 0 else None)

    for i, t in enumerate(temp_add):
        plt.axvline(x=t, ls='--', c='r', label='template_added' if i == 0 else None)

    plt.xlabel('Days since start')
    plt.ylabel('Log ' + title)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()