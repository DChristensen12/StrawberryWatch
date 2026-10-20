import numpy as np
import torch
from torch.amp import GradScaler, autocast

from strawberrywatch.config import Config
from strawberrywatch.models import model_calls


def _masked_mse(predictions, target, target_mask, scored_feature_idx=None):
    """
    MSE that skips cells where target_mask (batch, num_nodes) is False, and
    only compares the channels in scored_feature_idx when given. Channels are
    narrowed before the mask. A bad node is zeroed the same way however many
    are scored. With both None it's exactly nn.MSELoss()(predictions, target).
    """
    if scored_feature_idx is not None:
        predictions = predictions[..., scored_feature_idx]
        target = target[..., scored_feature_idx]
    squared_error = (predictions - target) ** 2
    if target_mask is None:
        return squared_error.mean()
    mask = target_mask.unsqueeze(-1).float()  # (batch, num_nodes, 1)
    denom = mask.sum() * squared_error.shape[-1]
    denom = denom.clamp(min=1.0)
    return (squared_error * mask).sum() / denom


def train_temporal_gnn(
    model,
    train_sequences,
    train_targets,
    edge_index,
    val_sequences=None,
    val_targets=None,
    feature_cols=None,
    train_node_mask=None,
    val_node_mask=None,
    train_target_mask=None,
    val_target_mask=None,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.PATIENCE,
    device=Config.DEVICE,
):
    """
    Trains a temporal GNN on MSE reconstruction. Returns (train_losses,
    val_losses, threshold, node_error_stats), with the last three empty or None
    without validation data.

    Mixed precision on CUDA, plain fp32 on CPU, since autocast breaks oneDNN's
    LSTM kernel there and doesn't make it faster.

    The node masks, (N, seq_len, num_nodes), go straight to the model, which
    fills in and ignores bad nodes. The target masks, (N, num_nodes), keep bad
    targets (like oxford mirroring university_house) out of the loss and the
    threshold. The model isn't taught to reproduce bad data. All four None
    is the old unmasked behavior.

    The threshold comes from validation errors on conductivity only ("Model
    All, Alert One"), or the mean over all features if feature_cols has no
    conductivity. node_error_stats is {node: {"median", "iqr", "threshold",
    "cond_median", "cond_iqr"}}: each node's error spread for per-node
    thresholds, plus the spread of its conductivity level, since a sustained
    step gets predicted well again after one step and only shows in the level.
    """
    model = model.to(device)
    edge_index = edge_index.to(device)

    # Not every model takes node_mask, and passing it to one that doesn't is a
    # TypeError, not a no-op. Ask once.
    supports_node_mask = model_calls.accepts_node_mask(model)
    if (train_node_mask is not None or val_node_mask is not None) and not supports_node_mask:
        print(
            f"node_mask data was provided but {type(model).__name__}.forward() doesn't "
            f"accept node_mask, QC masking will affect the loss/threshold calibration "
            f"below but NOT feature propagation or pooling inside the model."
        )

    def _run_model(seq_tensor, mask_tensor):
        return model_calls.run_sequence_model(
            model, seq_tensor, edge_index, mask_tensor if supports_node_mask else None
        )

    # The loss only scores Config.SCORED_TARGET_FEATURES. Weather and time
    # encodings stay inputs, and the output layer still predicts all of them.
    scored_feature_idx = None
    if feature_cols:
        scored_feature_idx = [
            feature_cols.index(f) for f in Config.SCORED_TARGET_FEATURES if f in feature_cols
        ]
        if scored_feature_idx:
            print(
                f"loss scored on: {[feature_cols[i] for i in scored_feature_idx]} "
                f"(indices {scored_feature_idx}), everything else is input-only"
            )
        else:
            print(
                "none of Config.SCORED_TARGET_FEATURES are in feature_cols, "
                "falling back to scoring all features"
            )
            scored_feature_idx = None

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    use_amp = device_type == "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler(device_type) if use_amp else None

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"training on {device_type.upper()} ({'mixed-precision' if use_amp else 'fp32'})...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(train_sequences), batch_size):
            batch_seq = torch.FloatTensor(train_sequences[i : i + batch_size]).to(device)
            batch_target = torch.FloatTensor(train_targets[i : i + batch_size]).to(device)
            batch_node_mask = None
            if train_node_mask is not None:
                batch_node_mask = torch.BoolTensor(train_node_mask[i : i + batch_size]).to(device)
            batch_target_mask = None
            if train_target_mask is not None:
                batch_target_mask = torch.BoolTensor(train_target_mask[i : i + batch_size]).to(
                    device
                )

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device_type):
                    predictions = _run_model(batch_seq, batch_node_mask)
                    loss = _masked_mse(
                        predictions, batch_target, batch_target_mask, scored_feature_idx
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = _run_model(batch_seq, batch_node_mask)
                loss = _masked_mse(predictions, batch_target, batch_target_mask, scored_feature_idx)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        if val_sequences is not None:
            model.eval()
            with torch.no_grad():
                val_seq = torch.FloatTensor(val_sequences).to(device)
                val_tgt = torch.FloatTensor(val_targets).to(device)
                val_seq_mask = (
                    torch.BoolTensor(val_node_mask).to(device)
                    if val_node_mask is not None
                    else None
                )
                val_tgt_mask = (
                    torch.BoolTensor(val_target_mask).to(device)
                    if val_target_mask is not None
                    else None
                )
                if use_amp:
                    with autocast(device_type=device_type):
                        val_pred = _run_model(val_seq, val_seq_mask)
                        val_loss = _masked_mse(
                            val_pred, val_tgt, val_tgt_mask, scored_feature_idx
                        ).item()
                else:
                    val_pred = _run_model(val_seq, val_seq_mask)
                    val_loss = _masked_mse(
                        val_pred, val_tgt, val_tgt_mask, scored_feature_idx
                    ).item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"early stopping at epoch {epoch + 1}")
                    break

        # Every epoch, flushed, to show progress while it trains
        status = f"Epoch {epoch + 1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f}"
        if val_sequences is not None:
            status += f" | Val Loss: {val_loss:.6f}"
        print(status, flush=True)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"restored best weights (val loss {best_val_loss:.6f})")

    # Conductivity-only threshold. Other channels add noise; spills show as
    # conductivity deviations. The model scores three and alerts on one.
    threshold = None
    node_error_stats = None
    if val_sequences is not None:
        model.eval()
        with torch.no_grad():
            val_seq = torch.FloatTensor(val_sequences).to(device)
            val_tgt = torch.FloatTensor(val_targets).to(device)
            val_seq_mask = (
                torch.BoolTensor(val_node_mask).to(device) if val_node_mask is not None else None
            )
            val_pred = _run_model(val_seq, val_seq_mask)
            val_errors = (val_pred - val_tgt).abs().cpu().numpy()
            val_tgt_np = val_tgt.cpu().numpy()

            # Masked targets give made-up errors. Skip them here like the
            # per-node loop does further down
            node_keep = (
                np.asarray(val_target_mask).astype(bool)
                if val_target_mask is not None
                else np.ones(val_errors.shape[:2], dtype=bool)
            )

            def _masked_node_mean(per_node):
                kept = node_keep.sum(axis=1)
                total = np.where(node_keep, per_node, 0.0).sum(axis=1)
                return np.divide(total, kept, out=np.full(len(total), np.nan), where=kept > 0)

            if feature_cols and "conductivity" in feature_cols:
                cond_idx = feature_cols.index("conductivity")
                # Average conductivity error across nodes per timestep.
                # System-wide spills score higher than single-site blips.
                system_scores = _masked_node_mean(val_errors[:, :, cond_idx])
                per_node_scores = val_errors[:, :, cond_idx]  # (n_val, num_nodes), node axis kept
                # the actual normalized conductivity value, not the error. This
                # is the level-shift anchor, kept separate from per_node_scores
                per_node_cond_level = val_tgt_np[:, :, cond_idx]  # (n_val, num_nodes)
                scoring_note = f"conductivity only (feature index {cond_idx})"
            else:
                # Fallback if feature_cols not passed in
                system_scores = _masked_node_mean(val_errors.mean(axis=2))
                per_node_scores = val_errors.mean(axis=2)  # (n_val, num_nodes)
                per_node_cond_level = None
                scoring_note = "mean across all features (fallback)"

            # A timestep with no valid node anywhere has no score, and can't
            # vote on the threshold either.
            usable_scores = system_scores[np.isfinite(system_scores)]
            n_dropped = len(system_scores) - len(usable_scores)
            if n_dropped:
                print(
                    f"  global threshold: excluding {n_dropped}/{len(system_scores)} "
                    f"timesteps with no valid target at any node"
                )
            if len(usable_scores) == 0:
                usable_scores = system_scores
            threshold = float(np.percentile(usable_scores, Config.THRESHOLD_PERCENTILE))

            # Median/IQR per node. A site that runs hotter (south_fork_2) is
            # judged against its own spread. QC-invalid targets are left out, as
            # HadISD does, or bad data would widen a node's "normal".
            num_nodes = per_node_scores.shape[1]
            node_error_stats = {}
            for node_idx in range(num_nodes):
                node_errs = per_node_scores[:, node_idx]
                node_level = (
                    per_node_cond_level[:, node_idx] if per_node_cond_level is not None else None
                )
                if val_target_mask is not None:
                    keep = val_target_mask[:, node_idx].astype(bool)
                    n_excluded = int((~keep).sum())
                    if keep.any():
                        node_errs = node_errs[keep]
                        if node_level is not None:
                            node_level = node_level[keep]
                    else:
                        print(
                            f"  node {node_idx}: every validation target masked, "
                            f"falling back to unfiltered errors for calibration"
                        )
                    if n_excluded:
                        print(
                            f"  node {node_idx}: excluding {n_excluded}/{len(keep)} "
                            f"QC-invalid targets from threshold calibration"
                        )
                median = float(np.median(node_errs))
                q75, q25 = np.percentile(node_errs, [75, 25])
                iqr = float(q75 - q25)
                # a node with a nearly flat error distribution can have IQR
                # basically zero, don't let that turn into a divide by zero
                safe_iqr = iqr if iqr > 1e-6 else 1.0
                normalized = (node_errs - median) / safe_iqr
                node_threshold = float(np.percentile(normalized, Config.THRESHOLD_PERCENTILE))

                # same robust median/IQR, but on the node's conductivity LEVEL,
                # the anchor Rule 2 (level shift) needs. See docstring above
                if node_level is not None and len(node_level) > 0:
                    cond_median = float(np.median(node_level))
                    cq75, cq25 = np.percentile(node_level, [75, 25])
                    cond_iqr = float(cq75 - cq25)
                    cond_iqr = cond_iqr if cond_iqr > 1e-6 else 1.0
                else:
                    cond_median, cond_iqr = None, None

                node_error_stats[node_idx] = {
                    "median": median,
                    "iqr": safe_iqr,
                    "threshold": node_threshold,
                    "cond_median": cond_median,
                    "cond_iqr": cond_iqr,
                }

        print(
            f"spill threshold: {threshold:.6f} "
            f"(P{Config.THRESHOLD_PERCENTILE}, scored on {scoring_note})"
        )

    print("training done.\n")
    return train_losses, val_losses, threshold, node_error_stats
