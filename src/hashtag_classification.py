from twitter_crawl import TwitterCrawl

class HtagClassifier:
    def __init__(self):
        pass

    @staticmethod
    def htag_classifier(list_json_frame):
        """
        :param list containing json dictionary:
        :return:
        """
        topic_dict = dict([])
        topic_dict["no_htags_tweet"] = []
        for json_frame in list_json_frame:
            no_htag_test = True
            for word in json_frame["text"].split(" "):
                if word.startswith("#"):
                    no_htag_test = False
                    try:
                        topic_dict[word].append((json_frame["text"], json_frame["date"]))
                    except:
                        topic_dict[word] = [(json_frame["text"], json_frame["date"])]
            if no_htag_test:
                topic_dict["no_htags_tweet"].append((json_frame["text"], json_frame["date"]))
        return topic_dict

    @staticmethod
    def get_most_frequent_htag(dic):
        lst = [word for word, frequency in sorted(dic.items(), key=lambda t: len(t[1]), reverse=True)]
        print("Most frequent tweets:", lst[1:6])
        return lst
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, MONDAY
from datetime import datetime, timedelta
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


from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib

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
        ax[ind].set_title('evolution of Htag #'+str(ind+1))
        ax[ind].autoscale_view()
        #ax.xaxis.grid(False, 'major')
        #ax.xaxis.grid(True, 'minor')
        ax[ind].grid(True)


        fig.autofmt_xdate()




    fig.subplots_adjust(hspace=0.5)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.show()


def plot(dic):
    dic = HtagClassifier().htag_classifier(dic)
    s = HtagClassifier().get_most_frequent_htag(dic)[1:6]


    time = []
    k = len(s)
    for i in range(k):
        time.append([])

    for htag in s:

        for j in range(len(dic[htag])):
            term = dic[htag][j][1]
            date = term[:19] + term[25:]
            time[s.index(htag)].append(datetime.strptime(date, "%a %b %d %H:%M:%S %Y"))

    nb_tweets_month = []

    duration = []
    nb_month = []

    for ind in range(k):

        duration.append(time[ind][0]-time[ind][-1])
        nb_month.append(monthdelta(time[ind][-1], time[ind][0]))
        nb_tweets_month.append([])

        for k in range(nb_month[ind]+1):
            nb_tweets_month[ind].append(0)

        for j in xrange(len(time[ind])-1, -1, -1):
            nb_tweets_month[ind][monthdelta(time[ind][-1], time[ind][j])] += 1


    evolution_all_month(time, nb_month, nb_tweets_month)

