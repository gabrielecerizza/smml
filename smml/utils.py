import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['cmr10']
plt.rc('text', usetex=True)


def heatmap(data, row_labels, col_labels, title, 
            xlabel, ylabel, ax=None,
            cbar_kw=None, cbarlabel='', **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Adapted from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.invert_yaxis()
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom')

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', top=False, left=False, bottom=False)

    ax.set_xlabel(xlabel, labelpad=15)
    ax.set_ylabel(ylabel, labelpad=15)
    ax.set_title(title, pad=15, weight='bold')

    return im, cbar


def annotate_heatmap(im, data=None, valfmt='{x:.3f}',
                     textcolors=('black', 'white'),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Adapted from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment='center',
              verticalalignment='center')
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_heatmap(
        data, param_grid, title, xlabel, ylabel, cbarlabel, 
        filename, cmap='viridis_r'):
    param_values = list(param_grid.values())
    data = data.reshape(
        len(param_values[0]), len(param_values[1]))
    fig, ax = plt.subplots()
    
    im, cbar = heatmap(data, param_values[0], param_values[1], 
                       ax=ax, cmap=cmap, 
                       cbarlabel=cbarlabel,
                       title=title, xlabel=xlabel, ylabel=ylabel)
    texts = annotate_heatmap(im, valfmt='{x:.3f}')

    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def plot_class_counts(y):
    classes, class_counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots()
    bars = ax.bar(classes, height=class_counts,
            tick_label=classes)
    bar_color = bars[0].get_facecolor()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            r'\textbf{' + str(round(bar.get_height(), 1)) + r'}',
            horizontalalignment='center',
            color=bar_color,
            weight='bold'
        )

    ax.set_xlabel('Class label', labelpad=15)
    ax.set_ylabel('Frequency', labelpad=15)
    ax.set_title(r'\textbf{Number of examples per class}', 
                 pad=15, weight='bold')
    fig.tight_layout()
    plt.savefig('img/class_counts.png', dpi=300, bbox_inches='tight')


def plot_tsne_data(X, y):
    X_embedded = TSNE(n_components=2, learning_rate='auto', 
                      init='random', perplexity=3).fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], 
            c=y, edgecolors='black', linewidths=0.5, cmap='viridis')
    ax.set_title(r'\textbf{Data visualization with t-SNE}', 
                    pad=15, weight='bold')
    fig.tight_layout()
    plt.savefig('img/tsne.png', dpi=300, bbox_inches='tight')


def plot_digits(X, y, n):
    plt.figure(figsize=(4, 5))
    for i in range(n):
        plt.subplot(n // 2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i].reshape((16, 16)), cmap=plt.cm.binary)
        plt.xlabel([y[i]])
    plt.tight_layout()
    plt.savefig('img/digits.png', dpi=300, bbox_inches='tight')


def plot_runtime_comparison(Ts, times):
    fig, ax = plt.subplots()

    ax.plot(Ts, times['naive'], label='naive')
    ax.plot(Ts, times['optimized'], label='optimized')
    ax.legend()
    ax.set_xlabel('Number of iterations (T)', labelpad=15)
    ax.set_ylabel('Runtime (s)', labelpad=15)
    ax.set_title(r'\textbf{Naive vs optimized implementation comparison}', 
                    pad=15, weight='bold')
    fig.tight_layout()
    plt.savefig('img/runtime.png', dpi=300, bbox_inches='tight')


def plot_cv_runtime(files, labels, param_grid):
    fig, ax = plt.subplots()
    Ts = param_grid['T']

    for file, label in zip(files, labels):
        res = ''
        with open(file,'r') as f:
            for i in f.readlines():
                res=i
        res = eval(res)

        times = np.array([val['time'] for val in res.values()])
        times = times.reshape((len(Ts), len(param_grid['l'])))
        times = times.mean(axis=-1)
        ax.plot(Ts, times, label=label)

    ax.legend()
    ax.set_xlabel('Number of iterations (T)', labelpad=15)
    ax.set_ylabel('Runtime (s)', labelpad=15)
    ax.set_title(r'\textbf{Average cross-validation runtime}', 
                    pad=15, weight='bold')
    fig.tight_layout()
    plt.savefig(
        'img/cv_runtime.png', dpi=300, bbox_inches='tight')
