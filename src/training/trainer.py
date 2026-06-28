import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from config.config import Config


def train_temporal_gnn(
    model,
    train_sequences,
    train_targets,
    edge_index,
    val_sequences=None,
    val_targets=None,
    feature_cols=None,
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

    After training, computes a spill-detection threshold from validation errors
    on conductivity only ("Model All, Alert One"). Threshold gets computed at
    train time so inference isn't sensitive to what it happens to see live.
    If feature_cols isn't provided or doesn't include conductivity, falls back
    to mean error across all features.

    Returns (train_losses, val_losses, threshold). val_losses and threshold are
    empty/None if no validation data was given.
    """
    model = model.to(device)
    edge_index = edge_index.to(device)

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    use_amp = device_type == "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler = GradScaler(device_type) if use_amp else None

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"--- Starting Training on {device_type.upper()} "
          f"({'mixed-precision' if use_amp else 'fp32'}) ---")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(train_sequences), batch_size):
            batch_seq = torch.FloatTensor(train_sequences[i:i+batch_size]).to(device)
            batch_target = torch.FloatTensor(train_targets[i:i+batch_size]).to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device_type):
                    predictions = model(
                        batch_seq,
                        edge_index,
                        batch_size=len(batch_seq),
                        num_nodes=batch_seq.shape[2],
                    )
                    loss = criterion(predictions, batch_target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(
                    batch_seq,
                    edge_index,
                    batch_size=len(batch_seq),
                    num_nodes=batch_seq.shape[2],
                )
                loss = criterion(predictions, batch_target)
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
                if use_amp:
                    with autocast(device_type=device_type):
                        val_pred = model(
                            val_seq,
                            edge_index,
                            batch_size=len(val_seq),
                            num_nodes=val_seq.shape[2],
                        )
                        val_loss = criterion(val_pred, val_tgt).item()
                else:
                    val_pred = model(
                        val_seq,
                        edge_index,
                        batch_size=len(val_seq),
                        num_nodes=val_seq.shape[2],
                    )
                    val_loss = criterion(val_pred, val_tgt).item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[STATUS] Early stopping triggered at epoch {epoch+1}")
                    break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            status = f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f}"
            if val_sequences is not None:
                status += f" | Val Loss: {val_loss:.6f}"
            print(status)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[INFO] Restored model weights from epoch with Val Loss: {best_val_loss:.6f}")

    # Conductivity-only threshold. Other channels add noise; spills show as
    # conductivity deviations. See "Model All, Alert One" in the SCMG paper.
    threshold = None
    if val_sequences is not None:
        model.eval()
        with torch.no_grad():
            val_seq = torch.FloatTensor(val_sequences).to(device)
            val_tgt = torch.FloatTensor(val_targets).to(device)
            val_pred = model(
                val_seq,
                edge_index,
                batch_size=len(val_seq),
                num_nodes=val_seq.shape[2],
            )
            val_errors = (val_pred - val_tgt).abs().cpu().numpy()

            if feature_cols and "conductivity" in feature_cols:
                cond_idx = feature_cols.index("conductivity")
                # Average conductivity error across nodes per timestep.
                # System-wide spills score higher than single-site blips.
                system_scores = val_errors[:, :, cond_idx].mean(axis=1)
                scoring_note = f"conductivity only (feature index {cond_idx})"
            else:
                # Fallback if feature_cols not passed in
                system_scores = val_errors.mean(axis=(1, 2))
                scoring_note = "mean across all features (fallback)"

            threshold = float(
                np.percentile(system_scores, Config.THRESHOLD_PERCENTILE)
            )

        print(
            f"[INFO] Computed spill threshold from validation set: "
            f"{threshold:.6f} (P{Config.THRESHOLD_PERCENTILE}, "
            f"scored on {scoring_note})"
        )

    print("--- Training Complete ---\n")
    return train_losses, val_losses, threshold
