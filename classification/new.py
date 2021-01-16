# import matplotlib.pyplot as plt
# import matplotlib

# gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
# matplotlib.use('TKAgg', force=True)

# print(matplotlib.get_backend())

# import matplotlib
# matplotlib.use('Qt5Agg')
# #matplotlib.use('Qt4Agg')
# # print(matplotlib.rcParams['backend'])
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(range(20), range(20))

# plt.show()
import matplotlib.pyplot as plt

year = [1500, 1600, 1700, 1800, 1900, 2000]




fig = plt.figure()   # Initializes current figure
plt.plot(year)  # Adds to current figure

plt.show()  # Shows plot
fig.show()