#Useful Python packages
import datetime

def date1900(d):

    date = datetime.date(1900, 1, 1) + datetime.timedelta(int(d))

    print(date)

    return date