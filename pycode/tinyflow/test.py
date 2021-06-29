import datetime

a = datetime.datetime.now()
b = datetime.timedelta(minutes=1,seconds=20)
b = a+b
print((b-a).microseconds)
