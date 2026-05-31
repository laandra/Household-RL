### CELL 13: Function for Plotting Graphs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, datetime

"Funkcija za risanje grafov"
def plotMultiY(X, Y=None, X_label="X_os", Y_Label=None,
               legend=None, title="title",
               save=False, grid=True, save_pdf=False, show_title=False):

    if Y is None:
        Y = []
    if Y_Label is None:
        Y_Label = ["" for _ in range(max(1, len(Y)))]
    if legend is None:
        legend = ["" for _ in range(max(1, len(Y)))]

    barve = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    line_type = ["-", "--", ":", "-."]

    if show_title:
        plt.title(title)

    # Popravek dolžin Y_Label in legend
    if len(Y_Label) < len(Y):
        for _ in range(len(Y) - len(Y_Label)):
            Y_Label.append(Y_Label[0])

    if len(legend) < len(Y):
        for _ in range(len(Y) - len(legend)):
            legend.append(legend[0])

    # Popravek dolžin X in Y
    if len(Y) > 0:
        if len(X) < len(Y[0]):
            X += [X[-1]] * (len(Y[0]) - len(X))
        elif len(X) > len(Y[0]):
            X = X[:len(Y[0])]

    for i in range(len(Y) - 1):
        if len(Y[i + 1]) < len(Y[i]):
            Y[i + 1] += [Y[i + 1][-1]] * (len(Y[i]) - len(Y[i + 1]))
        elif len(Y[i + 1]) > len(Y[i]):
            Y[i + 1] = Y[i + 1][:len(Y[i])]

    # Mreža
    if grid:
        plt.grid(color="Black", linestyle="--", linewidth=0.5)
    else:
        plt.grid(False)

    # Risanje grafov
    for i in range(len(Y)):
        plt.plot(X, Y[i], color=barve[i], label=legend[i], linestyle=line_type[i % len(line_type)])
        plt.legend()
        if i > 0:
            if Y_Label[i] != Y_Label[i - 1]:
                plt.ylabel(Y_Label[i], color=barve[i])
                plt.tick_params(axis="y", colors=barve[i])
        else:
            plt.xlabel(X_label)
            plt.ylabel(Y_Label[i] if len(Y_Label) > 0 else "")

    # Preveri, ali so X datumi
    if all(isinstance(x, (date, datetime)) for x in X):
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        # Če ni datumska os, prikaži vsako n-to oznako
        if len(X) > 10:
            step = max(1, len(X) // 10)  # max 10 oznak
            plt.xticks(X[::step], rotation=45)

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()

    # Shranjevanje
    if save:
        plt.savefig(f"{title}.png")
    if save_pdf:
        plt.savefig(f"{title}.pdf")

    plt.show()