import inspect

import numpy as np
import torch
from torch.amp import autocast, GradScaler

from strawberrywatch.config import Config


def _masked_mse(predictions, target, target_mask, scored_feature_idx=None):
    """
    MSE, but cells where target_mask is False don't count, and only the
    channels in scored_feature_idx (if given) count at all. target_mask is
    None or (batch, num_nodes) bool; broadcasts across the feature dim.
    scored_feature_idx is None (score every channel) or a list of indices
    into the feature dim -- narrows which channels are compared BEFORE the
    mask is applied, so QC masking still zeroes out a bad node the same way
    regardless of how many channels are being scored.

    With both args None this reduces to exactly nn.MSELoss()(predictions,
    target) (mean over every element) -- same numbers, so passing neither
    doesn't change training at all.
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
    Trains a temporal GNN via MSE reconstruction and returns losses plus a threshold.

    Mixed-precision on CUDA, plain fp32 on CPU (autocast on CPU breaks oneDNN's
    LSTM kernel and gives no speedup since CPUs lack tensor cores).

    train_node_mask/val_node_mask: optional (N, seq_len, num_nodes) bool, QC
    validity per node per input timestep (see prepare_sequences_normalized's
    return_node_mask). Passed straight through to the model so feature
    propagation fills in masked nodes and pooling ignores them.

    train_target_mask/val_target_mask: optional (N, num_nodes) bool, QC
    validity at the prediction target. A masked node's target is a real-looking
    but untrustworthy value (e.g. oxford mirroring university_house) -- scoring
    the model on reconstructing it would just teach it to reproduce bad data,
    so it's excluded from the loss and from threshold calibration below.

    All four default to None, which reduces to exactly the old unmasked
    behavior (node_mask=None into the model, plain mean-over-everything loss).

    After training, computes a spill-detection threshold from validation errors
    on conductivity only ("Model All, Alert One"). Threshold gets computed at
    train time so inference isn't sensitive to what it happens to see live.
    If feature_cols isn't provided or doesn't include conductivity, falls back
    to mean error across all features.

    Returns (train_losses, val_losses, threshold, node_error_stats). val_losses,
    threshold, and node_error_stats are empty/None if no validation data was
    given. node_error_stats is a dict keyed by node index (0..num_nodes-1),
    each value {"median", "iqr", "threshold", "cond_median", "cond_iqr"}.
    median/iqr/threshold characterize that node's error spread, for per-node
    thresholding instead of one global cutoff. cond_median/cond_iqr are a
    separate thing: the robust center and spread of the node's actual
    normalized conductivity LEVEL (not its error), for detecting a level shift
    that the error-based stats can't see -- a sustained step gets predicted
    correctly again after one timestep, so its error goes quiet, but the level
    itself stays away from normal the whole time.
    """
    model = model.to(device)
    edge_index = edge_index.to(device)

    # Not every model in the registry takes node_mask (Dusk Crayfish doesn't).
    # Check once instead of assuming, so training the default model doesn't
    # crash on an unexpected kwarg the moment a caller passes masks in.
    supports_node_mask = "node_mask" in inspect.signature(model.forward).parameters
    if (train_node_mask is not None or val_node_mask is not None) and not supports_node_mask:
        print(
            f"node_mask data was provided but {type(model).__name__}.forward() doesn't "
            f"accept node_mask -- QC masking will affect the loss/threshold calibration "
            f"below but NOT feature propagation or pooling inside the model."
        )

    def _run_model(seq_tensor, mask_tensor):
        kwargs = dict(batch_size=len(seq_tensor), num_nodes=seq_tensor.shape[2])
        if supports_node_mask:
            kwargs["node_mask"] = mask_tensor
        return model(seq_tensor, edge_index, **kwargs)

    # Loss only scores Config.SCORED_TARGET_FEATURES -- everything else (weather,
    # time encodings) stays a model input but stops being something we penalize
    # the model for mispredicting. Output layer is untouched (still predicts all
    # of feature_cols); this only narrows what the loss looks at.
    scored_feature_idx = None
    if feature_cols:
        scored_feature_idx = [
            feature_cols.index(f) for f in Config.SCORED_TARGET_FEATURES if f in feature_cols
        ]
        if scored_feature_idx:
            print(f"loss scored on: {[feature_cols[i] for i in scored_feature_idx]} "
                  f"(indices {scored_feature_idx}) -- everything else is input-only")
        else:
            print("none of Config.SCORED_TARGET_FEATURES are in feature_cols, "
                  "falling back to scoring all features")
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

    print(f"training on {device_type.upper()} "
          f"({'mixed-precision' if use_amp else 'fp32'})...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(train_sequences), batch_size):
            batch_seq = torch.FloatTensor(train_sequences[i:i+batch_size]).to(device)
            batch_target = torch.FloatTensor(train_targets[i:i+batch_size]).to(device)
            batch_node_mask = None
            if train_node_mask is not None:
                batch_node_mask = torch.BoolTensor(train_node_mask[i:i+batch_size]).to(device)
            batch_target_mask = None
            if train_target_mask is not None:
                batch_target_mask = torch.BoolTensor(train_target_mask[i:i+batch_size]).to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device_type):
                    predictions = _run_model(batch_seq, batch_node_mask)
                    loss = _masked_mse(predictions, batch_target, batch_target_mask, scored_feature_idx)
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
                val_seq_mask = torch.BoolTensor(val_node_mask).to(device) if val_node_mask is not None else None
                val_tgt_mask = torch.BoolTensor(val_target_mask).to(device) if val_target_mask is not None else None
                if use_amp:
                    with autocast(device_type=device_type):
                        val_pred = _run_model(val_seq, val_seq_mask)
                        val_loss = _masked_mse(val_pred, val_tgt, val_tgt_mask, scored_feature_idx).item()
                else:
                    val_pred = _run_model(val_seq, val_seq_mask)
                    val_loss = _masked_mse(val_pred, val_tgt, val_tgt_mask, scored_feature_idx).item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            status = f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f}"
            if val_sequences is not None:
                status += f" | Val Loss: {val_loss:.6f}"
            print(status)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"restored best weights (val loss {best_val_loss:.6f})")

    # Conductivity-only threshold. Other channels add noise; spills show as
    # conductivity deviations. See "Model All, Alert One" in the SCMG paper.
    threshold = None
    node_error_stats = None
    if val_sequences is not None:
        model.eval()
        with torch.no_grad():
            val_seq = torch.FloatTensor(val_sequences).to(device)
            val_tgt = torch.FloatTensor(val_targets).to(device)
            val_seq_mask = torch.BoolTensor(val_node_mask).to(device) if val_node_mask is not None else None
            val_pred = _run_model(val_seq, val_seq_mask)
            val_errors = (val_pred - val_tgt).abs().cpu().numpy()
            val_tgt_np = val_tgt.cpu().numpy()

            if feature_cols and "conductivity" in feature_cols:
                cond_idx = feature_cols.index("conductivity")
                # Average conductivity error across nodes per timestep.
                # System-wide spills score higher than single-site blips.
                system_scores = val_errors[:, :, cond_idx].mean(axis=1)
                per_node_scores = val_errors[:, :, cond_idx]  # (n_val, num_nodes), node axis kept
                # the actual normalized conductivity value, not the error -- this
                # is the level-shift anchor, kept separate from per_node_scores
                per_node_cond_level = val_tgt_np[:, :, cond_idx]  # (n_val, num_nodes)
                scoring_note = f"conductivity only (feature index {cond_idx})"
            else:
                # Fallback if feature_cols not passed in
                system_scores = val_errors.mean(axis=(1, 2))
                per_node_scores = val_errors.mean(axis=2)  # (n_val, num_nodes)
                per_node_cond_level = None
                scoring_note = "mean across all features (fallback)"

            threshold = float(
                np.percentile(system_scores, Config.THRESHOLD_PERCENTILE)
            )

            # median/IQR per node so a site that just runs hotter than the rest
            # (south_fork_2, say) doesn't sit permanently above one shared
            # global threshold. Each node gets judged against its own spread.
            #
            # QC-invalid targets get excluded here too, same rule HadISD uses:
            # flagged data doesn't get a vote in threshold derivation. A node
            # scored partly against known-bad targets (e.g. oxford during the
            # university_house duplication) would have its "normal" error
            # spread pulled toward however wrong those targets happen to be,
            # silently miscalibrating the threshold for every future reading.
            num_nodes = per_node_scores.shape[1]
            node_error_stats = {}
            for node_idx in range(num_nodes):
                node_errs = per_node_scores[:, node_idx]
                node_level = per_node_cond_level[:, node_idx] if per_node_cond_level is not None else None
                if val_target_mask is not None:
                    keep = val_target_mask[:, node_idx].astype(bool)
                    n_excluded = int((~keep).sum())
                    if keep.any():
                        node_errs = node_errs[keep]
                        if node_level is not None:
                            node_level = node_level[keep]
                    else:
                        print(f"  node {node_idx}: every validation target masked, "
                              f"falling back to unfiltered errors for calibration")
                    if n_excluded:
                        print(f"  node {node_idx}: excluding {n_excluded}/{len(keep)} "
                              f"QC-invalid targets from threshold calibration")
                median = float(np.median(node_errs))
                q75, q25 = np.percentile(node_errs, [75, 25])
                iqr = float(q75 - q25)
                # a node with a nearly flat error distribution can have IQR
                # basically zero, don't let that turn into a divide by zero
                safe_iqr = iqr if iqr > 1e-6 else 1.0
                normalized = (node_errs - median) / safe_iqr
                node_threshold = float(np.percentile(normalized, Config.THRESHOLD_PERCENTILE))

                # same robust median/IQR, but on the node's conductivity LEVEL,
                # the anchor Rule 2 (level shift) needs -- see docstring above
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
