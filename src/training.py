import copy
import torch as th


def evaluate(model, d):
    # AUC et AP sur un split (val ou test). Le message passing utilise d.edge_index :
    # val  -> liens de train ; test -> liens de train + val 
    model.eval()
    with th.no_grad():
        z = model.encode(d.x, d.edge_index)
        return model.test(z, d.pos_edge_label_index, d.neg_edge_label_index)  # (auc, ap)


def train_model(model, train_data, val_data, epochs=200, variational=True, lr=0.01):
    # Entraine le modele sur train_data. A chaque epoch on mesure l'AUC sur la validation
    #  et on garde les poids du meilleur epoch (aucune information du test n'est utilisee).

    opt = th.optim.Adam(model.parameters(), lr=lr)
    best_auc, best_state, history = -1.0, None, []

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        if variational:
            loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
        loss.backward()
        opt.step()

        val_auc, _ = evaluate(model, val_data)
        history.append((loss.item(), val_auc))
        if val_auc > best_auc:
            best_auc, best_state = val_auc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return {"best_val_auc": best_auc, "history": history}


def test_model(model, test_data):
    """Evaluation finale sur le test (a appeler une seule fois, apres l'entrainement)."""
    return evaluate(model, test_data)
