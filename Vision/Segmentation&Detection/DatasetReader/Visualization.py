import matplotlib.pyplot as plt

def show_bar_graph(data):
    names = [item['name'] for item in data]
    counts = [item['count'] for item in data]

    plt.figure(figsize=(20, 10))
    plt.bar(names, counts)
    plt.xticks(rotation=90)
    plt.xlabel('Object Classes')
    plt.ylabel('Count')
    plt.title('Object Class Counts')
    plt.tight_layout()
    plt.show()