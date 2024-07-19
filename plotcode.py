import sys

import pandas as pd
import matplotlib.pyplot as plt

def plot_bar(df, x_col, y_col, xlabel, ylabel, title):
    plt.figure(figsize=(8, 4))
    means = df.groupby(x_col)[y_col].mean()
    means.plot(kind='bar', color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)
    plt.show()

def plot_line(df, x_col, y_col, xlabel, ylabel, title):
    plt.figure(figsize=(8, 4))
    plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', color='b')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()

def main():
    info = pd.read_csv(r'C:\Users\niiha\Downloads\Exercise00 - 8286\TestInfo.csv')
    result = pd.read_pickle(r'C:\Users\niiha\Downloads\Exercise00 - 8286\TestResults.pickle')
    merged_data = pd.merge(result, info, on='TestId')
    plot_bar(merged_data, 'Device', 'Time (ms)', 'Devices', 'Mean Time (ms)', 'Mean Time Comparison Across Devices')
    plot_bar(merged_data, 'Device', 'PeakMemory (MB)', 'Devices', 'Mean Peak Memory (MB)', 'Mean Memory Comparison Across Devices')
    plot_bar(merged_data, 'Optimised', 'Time (ms)', 'Optimized', 'Mean Time (ms)', 'Mean Time Comparison: Optimized vs Non-Optimized')
    plot_bar(merged_data, 'Optimised', 'PeakMemory (MB)', 'Optimized', 'Mean Peak Memory (MB)', 'Mean Memory Comparison: Optimized vs Non-Optimized')

if __name__ == "__main__":
    sys.exit(main())


"""
Analysis
Deliverables: 
1)	Bullet point summary of your findings. 
a.	TestId is unique and is tagged in info and result data
b.	The result data mapping with info data suggests, we are trying to evaluated the MobileNet and AlexNet performance using time and memory metric
c.	Optimized runs are utilizing less memory and completing faster as compared to ‘N’ as optimized with less peak memory, though there is are some exceptions:
i.	For device_0 and MLNetwork as ‘AlexNet’, with same thread count, below data suggest, the test running also affect memory utilization
1.	86(Peak Memory), Device_0, 1000(CPU Freq),5(Threads),MobileNet
2.	450(Peak Memory), Device_0, 1000(CPU Freq),5(Threads),MobileNet
d.	AlexNet’s peak memory is higher, seems the test system is having dependency on what CNN is being run on.
2)	Visualisations that helped you analyse the data with added annotations. 
a.	I tried pandas utils like read_csv and read_pickle and merged the data using common value which is TestID
b.	Matplotlib plotting helped visualize the conclusion.
3)	Follow up actions regarding your findings you would take with the development team
a.	MLNetwork should be debugged for test id having higher Peak Memory with optimization enabled.

"""