import os
import time
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Importdata import LoadData
from model import ASMSTCN_Reg


def compute_metrics(true_vals, pred_vals):
    y_true = true_vals.reshape(-1)
    y_pred = pred_vals.reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    if np.std(y_true) < 1e-6:
        r2 = 1.0 if mean_squared_error(y_true, y_pred) < 1e-12 else 0.0
    else:
        r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def train_epoch(data, model, criterion, optimizer,
                batch_size, space_loss_weight):
    model.train()
    device = next(model.parameters()).device
    for Xb, Yb in data.get_batches(data.train[0], data.train[1], batch_size, shuffle=True):
        Xb, Yb = Xb.to(device), Yb.to(device)
        optimizer.zero_grad()
        pred, loss_space = model(Xb)
        pred = pred.squeeze()
        y = Yb.squeeze()
        sup_loss = criterion(pred, y)
        loss = sup_loss + space_loss_weight * loss_space
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate_split(data, model, batch_size, which="valid"):
    model.eval()
    device = next(model.parameters()).device
    if which == "valid":
        X, Y = data.valid
    elif which == "test":
        X, Y = data.test
    elif which == "train":
        X, Y = data.train
    else:
        raise ValueError(f"Unknown split: {which}")

    preds, trues = [], []
    for Xb, Yb in data.get_batches(X, Y, batch_size, shuffle=False):
        Xb, Yb = Xb.to(device), Yb.to(device)
        out = model(Xb)
        pred = out[0] if isinstance(out, (tuple, list)) else out

        # Ensure (B,H,N)
        if pred.dim() == 4 and pred.size(-1) == 1:
            pred = pred.squeeze(-1)
        if Yb.dim() == 4 and Yb.size(-1) == 1:
            Yb = Yb.squeeze(-1)

        preds.append(pred.detach().cpu())
        trues.append(Yb.detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    return preds, trues


def run_training(data,
                 input_seq_len, output_seq_len,
                 learning_rate, batch_size,
                 epochs, hidden_dim,
                 space_loss_weight,
                 nmb_prototype, lap_reg_weight,
                 tau, tau_dyn, alpha_init,
                 early_stop_patience=30,
                 save_dir="./saved_models"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建保存模型的目录
    os.makedirs(save_dir, exist_ok=True)

    model = ASMSTCN_Reg(
        adj_matrix=data.adj,
        num_nodes=data.num_nodes,
        input_length=input_seq_len,
        input_dim=data.features,
        hidden_dim=hidden_dim,
        output_dim=1,
        output_length=output_seq_len,
        dropout=0.1,
        nmb_prototype=nmb_prototype,
        lap_reg_weight=lap_reg_weight,
        device=device,
        coordinates=data.coordinates,
        tau_dyn=tau_dyn,
        alpha_init=alpha_init
    ).to(device)

    if hasattr(model, "spatial_reg") and hasattr(model.spatial_reg, "tau"):
        model.spatial_reg.tau = torch.tensor(float(tau), device=device)

    criterion = nn.SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    best_val_mae = float('inf')
    best_state = None
    best_epoch = -1
    no_improve = 0

    best_model_path = os.path.join(save_dir, f"best_model_h{output_seq_len}_epoch{best_epoch}.pt")
    last_tested_best_epoch = -1
    for epoch in range(1, epochs + 1):
        train_epoch(data, model, criterion, optimizer,
                    batch_size, space_loss_weight)

        # ---- Validation ----
        v_p, v_t = evaluate_split(data, model, batch_size, which="valid")
        v_p = data.inverse_y(v_p)
        v_t = data.inverse_y(v_t)
        v_mae, v_rmse, v_r2 = compute_metrics(v_t, v_p)

        improved = False
        if v_mae < best_val_mae - 1e-8:
            best_val_mae = v_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
            improved = True

            # 保存最佳模型权重到文件
            best_model_path = os.path.join(save_dir, f"best_model_h{output_seq_len}_epoch{epoch}_mae{v_mae:.4f}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state,
                'val_mae': v_mae,
                'val_rmse': v_rmse,
                'val_r2': v_r2,
                'hyperparams': {
                    'input_seq_len': input_seq_len,
                    'output_seq_len': output_seq_len,
                    'hidden_dim': hidden_dim,
                    'space_loss_weight': space_loss_weight,
                    'lap_reg_weight': lap_reg_weight,
                    'tau_dyn': tau_dyn,
                    'alpha_init': alpha_init,
                }
            }, best_model_path)
            print(f"  [Saved] Best model saved to {best_model_path}")

        else:
            no_improve += 1

        print(f"Epoch {epoch:03d} | Val_MAE {v_mae:.4f} | Val_RMSE {v_rmse:.4f} | Val_R2 {v_r2:.4f} "
              f"| Best@{best_epoch} Val_MAE {best_val_mae:.4f}{' *' if improved else ''}")

        # ---- Every 5 epochs: TEST only if best updated since last test ----
        if epoch % 5 == 0:
            if best_state is None:
                print(f"  [TEST skipped @ epoch {epoch:03d}] best model not available yet.")
            elif best_epoch != last_tested_best_epoch:
                # temporarily load best weights, test, then restore current weights
                cur_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_state)

                t_p, t_t = evaluate_split(data, model, batch_size, which="test")
                t_p = data.inverse_y(t_p)
                t_t = data.inverse_y(t_t)
                t_mae, t_rmse, t_r2 = compute_metrics(t_t, t_p)

                print(f"  [TEST @ epoch {epoch:03d}] using BEST@{best_epoch} | "
                      f"MAE={t_mae:.4f} RMSE={t_rmse:.4f} R2={t_r2:.4f}")

                last_tested_best_epoch = best_epoch
                model.load_state_dict(cur_state)
            else:
                print(f"  [TEST skipped @ epoch {epoch:03d}] best model not updated (best_epoch={best_epoch}).")

        # ---- Early stopping ----
        if no_improve >= early_stop_patience:
            print(f"[EarlyStop] no improvement for {early_stop_patience} epochs. Stop at epoch {epoch}.")
            break

    # ---- Final test with best model ----
    if best_state is not None:
        model.load_state_dict(best_state)

        # 再次保存最终的最佳模型
        final_model_path = os.path.join(save_dir, f"final_best_model_h{output_seq_len}.pt")
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_state,
            'val_mae': best_val_mae,
            'hyperparams': {
                'input_seq_len': input_seq_len,
                'output_seq_len': output_seq_len,
                'hidden_dim': hidden_dim,
                'space_loss_weight': space_loss_weight,
                'lap_reg_weight': lap_reg_weight,
                'tau_dyn': tau_dyn,
                'alpha_init': alpha_init,
            }
        }, final_model_path)
        print(f"[Final] Best model saved to {final_model_path}")

    t_p, t_t = evaluate_split(data, model, batch_size, which="test")
    t_p = data.inverse_y(t_p)
    t_t = data.inverse_y(t_t)
    mae, rmse, r2 = compute_metrics(t_t, t_p)
    print(f"Best@epoch {best_epoch} | out={output_seq_len}h | Test: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

    # 返回最佳模型路径和指标
    return {
        'best_model_path': best_model_path,
        'best_epoch': best_epoch,
        'best_val_mae': best_val_mae,
        'test_mae': mae,
        'test_rmse': rmse,
        'test_r2': r2
    }


def run_all_experiments():
    out_lens = [24]
    nmb_prototype = 50
    lap_reg_weight = 0.01
    space_loss_weight = 0.5
    tau = 0.1
    tau_dyn = 1.2
    alpha_init = 0.1

    # 设置保存目录
    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # 记录所有实验的结果
    all_results = []

    for i, out_h in enumerate(out_lens):
        print(f"\n{'=' * 60}")
        print(f"Training for output horizon {out_h}h (Experiment {i + 1}/{len(out_lens)})")
        print(f"{'=' * 60}")

        missing_path = f"./missing_stats_h{out_h}h.csv"
        data = LoadData(0.6, 0.2, 72, out_h, max_impute_gap=3, verbose=True, missing_stats_path=missing_path)

        # 运行训练，返回结果
        result = run_training(
            data,
            input_seq_len=72,
            output_seq_len=out_h,
            learning_rate=1e-4,
            batch_size=32,
            epochs=100,
            hidden_dim=64,
            space_loss_weight=space_loss_weight,
            nmb_prototype=nmb_prototype,
            lap_reg_weight=lap_reg_weight,
            tau=tau,
            tau_dyn=tau_dyn,
            alpha_init=alpha_init,
            early_stop_patience=20,
            save_dir=save_dir  # 传递保存目录
        )
        all_results.append(result)

        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 打印所有实验的汇总结果
    print(f"\n{'=' * 60}")
    print("Summary of All Experiments:")
    print(f"{'=' * 60}")
    for i, res in enumerate(all_results):
        print(f"Exp {i + 1} (h={out_lens[i]}): "
              f"Best epoch={res['best_epoch']}, "
              f"Val MAE={res['best_val_mae']:.4f}, "
              f"Test MAE={res['test_mae']:.4f}, "
              f"Test R2={res['test_r2']:.4f}")
        print(f"  Model saved at: {res['best_model_path']}")

    # 保存所有结果到文件
    results_path = os.path.join(save_dir, "all_experiments_results.txt")
    with open(results_path, 'w') as f:
        f.write("All Experiments Results\n")
        f.write("=" * 60 + "\n")
        for i, res in enumerate(all_results):
            f.write(f"Experiment {i + 1} (h={out_lens[i]})\n")
            f.write(f"  Best epoch: {res['best_epoch']}\n")
            f.write(f"  Best validation MAE: {res['best_val_mae']:.4f}\n")
            f.write(f"  Test MAE: {res['test_mae']:.4f}\n")
            f.write(f"  Test RMSE: {res['test_rmse']:.4f}\n")
            f.write(f"  Test R2: {res['test_r2']:.4f}\n")
            f.write(f"  Model saved at: {res['best_model_path']}\n")
            f.write("-" * 40 + "\n")
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    run_all_experiments()