import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, MONDAY
from datetime import datetime, timedelta
import json
from calendar import monthrange
import numpy


def monthdelta(d1, d2):
    delta = 0
    while True:
        mdays = monthrange(d1.year, d1.month)[1]
        d1 += timedelta(days=mdays)
        if d1 <= d2:
            delta += 1
        else:
            break
    return delta

def evolution_all_month(time, nb_month, nb_tweets_month):

    # every monday
    mondays = WeekdayLocator(MONDAY)
    # every 3rd month
    months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
    monthsFmt = DateFormatter("%b '%y")
    k = len(time)
    fig, ax = plt.subplots(nrows=k, sharex=True)
    dates = []
    for ind in range(k):
        dates.append([])
        ndays = 0
        first_day = time[ind][-1]
        for dt in range(nb_month[ind]+1):
            month = first_day.month+dt
            if month % 13 == 0:
                ndays += monthrange(first_day.year+month/12, 1)[1]
            else:
                ndays += monthrange(first_day.year+month/12, month % 13)[1]

            dates[ind].append(timedelta(days=ndays)+first_day)


        ax[ind].bar(dates[ind], nb_tweets_month[ind], width=20, color=numpy.random.rand(3, 1))
        ax[ind].xaxis_date()
        ax[ind].xaxis.set_major_locator(months)
        ax[ind].xaxis.set_major_formatter(monthsFmt)
        ax[ind].xaxis.set_minor_locator(mondays)
        ax[ind].set_title('evolution of cluster #'+str(ind+1))
        ax[ind].autoscale_view()
        #ax.xaxis.grid(False, 'major')
        #ax.xaxis.grid(True, 'minor')
        ax[ind].grid(True)

    fig.autofmt_xdate()

    fig.subplots_adjust(hspace=0.5)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.show()


def polarity_evolution(time, nb_month, nb_tweets_P, nb_tweets_N):
    # every monday
    mondays = WeekdayLocator(MONDAY)
    # every 3rd month
    months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
    monthsFmt = DateFormatter("%b '%y")
    k = len(time)
    fig, ax = plt.subplots(nrows=k, sharex=True)
    dates = []
    for ind in range(k):
        dates.append([])
        ndays = 0
        first_day = time[ind][-1]
        for dt in range(nb_month[ind]+1):
            month = first_day.month+dt
            if month % 13 == 0:
                ndays += monthrange(first_day.year+month/12, 1)[1]
            else:
                ndays += monthrange(first_day.year+month/12, month % 13)[1]

            dates[ind].append(timedelta(days=ndays)+first_day)


        ax[ind].plot(dates[ind], nb_tweets_P[ind], c='b', label="Positive tweets")
        ax[ind].plot(dates[ind], nb_tweets_N[ind], c='r', label="Negative tweets")
        ax[ind].xaxis_date()
        ax[ind].xaxis.set_major_locator(months)
        ax[ind].xaxis.set_major_formatter(monthsFmt)
        ax[ind].xaxis.set_minor_locator(mondays)
        ax[ind].set_title('Polarity evolution of topic #'+str(ind+1))
        ax[ind].autoscale_view()
        #ax.xaxis.grid(False, 'major')
        #ax.xaxis.grid(True, 'minor')
        ax[ind].grid(True)

    fig.autofmt_xdate()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, 'upper right')
    fig.subplots_adjust(hspace=0.5)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.show()



def present_all_data(k, tweet_list):

    time = []
    polarity = []

    for i in range(k):
        time.append([])
        polarity.append([])

    for tweet in tweet_list:
        date = tweet["date"][:19] + tweet["date"][25:]
        time[tweet["topic"]].append(datetime.strptime(date, "%a %b %d %H:%M:%S %Y"))
        polarity[tweet["topic"]].append(tweet["polarity"])

    nb_tweets_month = []
    nb_tweets_P = []
    nb_tweets_N = []
    nb_month = []

    for ind in range(k):

        nb_month.append(monthdelta(time[ind][-1], time[ind][0]))
        nb_tweets_month.append([])
        nb_tweets_P.append([])
        nb_tweets_N.append([])

        for k in range(nb_month[ind]+1):
            nb_tweets_month[ind].append(0)
            nb_tweets_P[ind].append(0)
            nb_tweets_N[ind].append(0)

        for j in xrange(len(time[ind])-1, -1, -1):
            delta = monthdelta(time[ind][j], time[ind][0])
            nb_tweets_month[ind][delta] += 1
            if polarity[ind][j] == 'pos':
                nb_tweets_P[ind][delta] += 1
            else:
                nb_tweets_N[ind][delta] += 1


    evolution_all_month(time, nb_month, nb_tweets_month)
    polarity_evolution(time, nb_month, nb_tweets_P, nb_tweets_N)
