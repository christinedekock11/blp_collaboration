import pandas as pd
from dateutil import parser
import datetime
import numpy as np
import json
import matplotlib.pylab as plt

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
    edits = page_revisions[-2:].revision_text_bytes.values
    return edits[0] / edits[1]


def num_revisions(page_revisions):
    return len(page_revisions)


def num_editors(page_revisions):
    return page_revisions.event_user_id.nunique()


def frac_recent_revisions(page_revisions, days=TIMEFRAME_DAYS):
    start_time = page_revisions.event_timestamp.max() - datetime.timedelta(days=days)
    recent = page_revisions.query('event_timestamp > @start_time')
    return len(recent) / len(page_revisions)


# group by page and user

def num_revisions(user_revisions):
    return len(user_revisions)


def frac_edits_reverted(user_revisions):
    return user_revisions.revision_is_identity_reverted.mean()


def frac_reverted_others(user_revisions):
    return user_revisions.revision_is_identity_revert.mean()


def mean_revision_size(user_revisions):
    return user_revisions.edit_size.mean()


def size_of_edits_after(user_revisions):
    return user_revisions.next_edit_size.mean()


def time_to_respond(user_revisions):
    return user_revisions.time_to_respond.mean()


def time_reponded_to(user_revisions):
    return user_revisions.time_responded_to.mean()


## these are all just column means; easier to just do them at once

def user_revision_feat(user_revisions):
    return user_revisions[['revision_is_identity_reverted', 'revision_is_identity_revert', 'edit_size',
                           'next_edit_size', 'time_to_respond', 'time_responded_to']].mean()


def daily_activity(data, start_date):
    data['days_since_start'] = (data['event_timestamp'] - start_date).dt.days
    daily_edit_count = data.groupby('days_since_start').size()
    daily_edit_vol = data.groupby('days_since_start')['edit_size'].apply(lambda x: x.abs().sum())
    df = pd.DataFrame({'daily_edit_count': daily_edit_count, 'daily_edit_vol': daily_edit_vol})
    df = df.reindex(np.arange(df.index.min(), df.index.max() + 1)).fillna(0)

    return df


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