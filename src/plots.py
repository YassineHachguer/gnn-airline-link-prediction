import matplotlib.pyplot as plt


def plot_comparison(mean, std, ratio_label="10% caches", metric="AUC"):
    m = mean[(ratio_label, metric)]
    s = std[(ratio_label, metric)]
    plt.figure(figsize=(9, 6))
    plt.barh(range(len(m)), m.values, xerr=s.values, color="steelblue")
    plt.yticks(range(len(m)), [i.strip() for i in m.index])
    plt.gca().invert_yaxis()
    plt.xlim(max(0.0, m.min() - 0.1), 1.0)
    plt.xlabel(f"{metric} (test)")
    plt.title(f"Comparaison des methodes - {ratio_label}")
    plt.tight_layout()
    plt.show()


def plot_grid(mean, grid_names, convs, decs, dec_label, ratio_label="10% caches", metric="AUC"):
    mat = [[mean.loc[grid_names[(c, d)], (ratio_label, metric)] for d in decs] for c in convs]
    plt.figure(figsize=(6, 3.5))
    plt.imshow(mat, cmap="Blues")
    plt.xticks(range(len(decs)), [dec_label[d] for d in decs])
    plt.yticks(range(len(convs)), [c.upper() for c in convs])
    for i in range(len(convs)):
        for j in range(len(decs)):
            plt.text(j, i, f"{mat[i][j]:.3f}", ha="center", va="center")
    plt.colorbar(label=metric)
    plt.title(f"{metric} test - encodeur x decodeur ({ratio_label})")
    plt.tight_layout()
    plt.show()


def plot_training_curves(history):
    loss = [h[0] for h in history]
    val = [h[1] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(loss); ax[0].set_title("Loss (train)"); ax[0].set_xlabel("epoch")
    ax[1].plot(val); ax[1].set_title("AUC (validation)"); ax[1].set_xlabel("epoch")
    plt.tight_layout()
    plt.show()
    