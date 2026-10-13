import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from kan import KAN
import glob
import os
import warnings
from collections import Counter, deque, defaultdict
from tqdm import tqdm
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================
class Config:
    USE_FRAME_SAMPLING = True
    SAMPLE_RATE = 2

    USE_AUGMENTATION = True
    AUG_NOISE_LEVELS = [0.01, 0.02]
    AUG_TIME_WARP_FACTORS = [0.95]

    # ⭐ نفس إعداداتك السريعة
    WINDOW_SIZE = 40
    WINDOW_STRIDE = 15
    MAX_WINDOWS_PER_FILE = 10

    FEATURES = 14
    CLASSES = 9

    # ⭐ نموذج بسيط زيادة صغيرة فقط
    GRU_HIDDEN_SIZE = 38         # من 32 → 38 (زيادة طفيفة)
    GRU_NUM_LAYERS = 1
    ATTENTION_HEADS = 2          # 38/2 = 19 (يعمل)
    KAN_GRID_SIZE = 3
    KAN_SPLINE_ORDER = 3
    KAN_WIDTH = [38, 19, 9]      # نفس النمط

    # ⭐⭐⭐ Focal Loss (بديل CrossEntropy)
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25

    # ⭐ Hyperparameters كما في كودك
    LEARNING_RATE = 0.0004
    BATCH_SIZE = 64
    EPOCHS = 60
    DROPOUT_RATE = 0.4           #
    WEIGHT_DECAY = 5e-4
    PATIENCE = 15

    MIXUP_ALPHA = 0.15

    # ⭐ Scheduler مع Warmup
    WARMUP_EPOCHS = 5
    COSINE_T0 = 10
    COSINE_T_MULT = 2
    COSINE_ETA_MIN = 1e-5

    SMOOTHING_WINDOW = 5

    DATA_FOLDER = r"A:\leapc-python-bindings-main\examples\my_hand_data"
    MODEL_SAVE_PATH = "hand_gesture_fast_smart_9classes.pth"

    TEST_FILES_PER_GESTURE = 10

    REQUIRED_FEATURES = [
        'palm_x', 'palm_y', 'palm_z',
        'palm_velocity_x', 'palm_velocity_y', 'palm_velocity_z',
        'grab_strength', 'pinch_strength', 'pinch_distance',
        'thumb_extended', 'index_extended', 'middle_extended',
        'ring_extended', 'pinky_extended'
    ]


# ==================== FOCAL LOSS ====================
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


# ==================== DATA AUGMENTATION ====================
class DataAugmentor:
    @staticmethod
    def add_noise(seq, noise_level):
        return seq + np.random.normal(0, noise_level, seq.shape)

    @staticmethod
    def time_warp(seq, factor):
        orig_len = seq.shape[0]
        new_len = int(orig_len * factor)
        indices = np.linspace(0, orig_len - 1, new_len)
        warped = np.zeros((new_len, seq.shape[1]))
        for f in range(seq.shape[1]):
            interp = interp1d(np.arange(orig_len), seq[:, f], kind='linear', fill_value='extrapolate')
            warped[:, f] = interp(indices)
        final_indices = np.linspace(0, new_len - 1, orig_len)
        result = np.zeros((orig_len, seq.shape[1]))
        for f in range(seq.shape[1]):
            interp = interp1d(np.arange(new_len), warped[:, f], kind='linear', fill_value='extrapolate')
            result[:, f] = interp(final_indices)
        return result

    @staticmethod
    def augment_sequence(seq):
        """Original + 2 Noise + 1 TimeWarp = 4 نسخ (نفس كودك)"""
        augmented = [seq]
        if not Config.USE_AUGMENTATION:
            return augmented
        for nl in Config.AUG_NOISE_LEVELS:
            augmented.append(DataAugmentor.add_noise(seq, nl))
        for tf in Config.AUG_TIME_WARP_FACTORS:
            if tf != 1.0:
                augmented.append(DataAugmentor.time_warp(seq, tf))
        return augmented


# ==================== SLIDING WINDOW ====================
class SlidingWindowProcessor:
    def __init__(self, window_size=40, stride=15, max_windows=10):
        self.window_size = window_size
        self.stride = stride
        self.max_windows = max_windows

    def create_windows(self, sequence):
        total_frames = sequence.shape[0]
        if total_frames < self.window_size:
            padding = np.zeros((self.window_size - total_frames, sequence.shape[1]))
            return [np.vstack([sequence, padding])]
        windows = []
        num_possible = (total_frames - self.window_size) // self.stride + 1
        if self.max_windows and num_possible > self.max_windows:
            indices = np.linspace(0, num_possible - 1, self.max_windows, dtype=int)
            for i in indices:
                start = i * self.stride
                windows.append(sequence[start:start + self.window_size])
        else:
            for start in range(0, total_frames - self.window_size + 1, self.stride):
                windows.append(sequence[start:start + self.window_size])
        return windows


# ==================== MODEL (مع Attention Pooling + Residual) ====================
class HandGestureModel(nn.Module):
    def __init__(self, input_features=14, num_classes=9, gru_hidden_size=38):
        super().__init__()
        assert Config.KAN_WIDTH[0] == gru_hidden_size

        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=gru_hidden_size,
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            dropout=0
        )
        self.layer_norm = nn.LayerNorm(input_features)

        self.input_dropout = nn.Dropout(0.15)
        self.gru_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        self.attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size,
            num_heads=Config.ATTENTION_HEADS,
            dropout=Config.DROPOUT_RATE,
            batch_first=True
        )

        # ⭐⭐⭐ Attention Pooling (بدل Mean Pooling)
        # يتعلم وزن كل إطار زمني بدلاً من المتوسط البسيط
        self.pool_attention = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(gru_hidden_size // 2, 1)
        )

        # ⭐ Residual Projection قبل KAN
        self.residual_proj = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size),
            nn.LayerNorm(gru_hidden_size),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )

        self.kan = KAN(
            width=Config.KAN_WIDTH,
            grid=Config.KAN_GRID_SIZE,
            k=Config.KAN_SPLINE_ORDER,
            seed=42
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.input_dropout(x)

        gru_out, _ = self.gru(x)
        gru_out = self.gru_dropout(gru_out)

        attn_out, _ = self.attention(gru_out, gru_out, gru_out)

        # ⭐ Attention Pooling (weighted sum)
        weights = self.pool_attention(attn_out)         # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        context = (attn_out * weights).sum(dim=1)       # (B, H)

        context = self.dropout(context)

        # ⭐ Residual
        context = context + self.residual_proj(context)

        return self.kan(context)


# ==================== DATA PROCESSOR ====================
class DataProcessor:
    def __init__(self):
        self.window_processor = SlidingWindowProcessor(
            window_size=Config.WINDOW_SIZE,
            stride=Config.WINDOW_STRIDE,
            max_windows=Config.MAX_WINDOWS_PER_FILE
        )
        self.scaler = None

    def extract_gesture_name(self, filename):
        name = os.path.basename(filename).lower()
        gestures = ['open', 'close', 'grab', 'push', 'pull',
                    'raise', 'lower', 'left', 'right']
        for gesture in gestures:
            if gesture in name:
                return gesture
        return None

    def load_single_file_with_augment(self, file_path):
        try:
            df = pd.read_csv(file_path)
            gesture = self.extract_gesture_name(os.path.basename(file_path))
            if not gesture:
                return None, None
            for f in Config.REQUIRED_FEATURES:
                if f not in df.columns:
                    df[f] = 0.0
            data = df[Config.REQUIRED_FEATURES].values.astype(np.float32)
            if Config.USE_FRAME_SAMPLING:
                data = data[::Config.SAMPLE_RATE]
            sequences = DataAugmentor.augment_sequence(data)
            return sequences, gesture
        except Exception as e:
            print(f"⚠️ Error {file_path}: {e}")
            return None, None

    def load_all_files_by_gesture(self, data_folder):
        csv_files = glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True)
        if not csv_files:
            raise ValueError(f"No CSV files in {data_folder}")
        print("\n🔍 Sample filenames:")
        for fname in csv_files[:5]:
            print(f"   {os.path.basename(fname)}")

        file_data = []
        gesture_counts = Counter()
        for fpath in tqdm(csv_files, desc="Loading files"):
            seqs, gesture = self.load_single_file_with_augment(fpath)
            if seqs and gesture:
                file_data.append((seqs, gesture, fpath))
                gesture_counts[gesture] += 1
        print("\n📊 Detected gesture counts:")
        for g, cnt in sorted(gesture_counts.items()):
            print(f"   {g}: {cnt} files")
        return file_data

    def fit_scaler_on_train(self, train_windows):
        n_w, win_len, n_f = train_windows.shape
        flat = train_windows.reshape(-1, n_f)
        self.scaler = StandardScaler()
        self.scaler.fit(flat)
        print(f"✅ Scaler fitted on {flat.shape[0]} samples")

    def normalize_windows(self, windows):
        shape = windows.shape
        flat = windows.reshape(-1, shape[-1])
        return self.scaler.transform(flat).reshape(shape)


# ==================== PREPARE DATA ====================
def prepare_data():
    print("=" * 60)
    print("📂 Loading data")
    processor = DataProcessor()
    file_data = processor.load_all_files_by_gesture(Config.DATA_FOLDER)

    if not file_data:
        raise ValueError("No valid data files found!")

    files_by_gesture = defaultdict(list)
    for item in file_data:
        files_by_gesture[item[1]].append(item)

    rng = np.random.RandomState(42)
    test_files, remaining_files = [], []
    for gesture, files in files_by_gesture.items():
        idx = rng.permutation(len(files))
        files_shuffled = [files[i] for i in idx]
        n_test = min(Config.TEST_FILES_PER_GESTURE, max(1, len(files_shuffled) // 3))
        test_files.extend(files_shuffled[:n_test])
        remaining_files.extend(files_shuffled[n_test:])

    remaining_labels = [g for (_, g, _) in remaining_files]
    train_files, val_files = train_test_split(
        remaining_files, test_size=0.2,
        stratify=remaining_labels, random_state=42
    )

    total = len(train_files) + len(val_files) + len(test_files)
    print(f"\n✅ Split summary:")
    print(f"   Train = {len(train_files)} files ({100 * len(train_files) / total:.1f}%)")
    print(f"   Val   = {len(val_files)} files ({100 * len(val_files) / total:.1f}%)")
    print(f"   Test  = {len(test_files)} files ({100 * len(test_files) / total:.1f}%)")

    def extract_windows(file_list, augment_train):
        all_windows, all_labels = [], []
        for sequences, gesture, _ in file_list:
            seqs_to_use = sequences if (augment_train and Config.USE_AUGMENTATION) else [sequences[0]]
            for seq in seqs_to_use:
                wins = processor.window_processor.create_windows(seq)
                for w in wins:
                    all_windows.append(w)
                    all_labels.append(gesture)
        return np.array(all_windows, dtype=np.float32), np.array(all_labels)

    X_train_raw, y_train = extract_windows(train_files, augment_train=True)
    X_val_raw, y_val = extract_windows(val_files, augment_train=False)
    X_test_raw, y_test = extract_windows(test_files, augment_train=False)

    print(f"\nRaw windows: Train {X_train_raw.shape}, Val {X_val_raw.shape}, Test {X_test_raw.shape}")
    processor.fit_scaler_on_train(X_train_raw)

    X_train = processor.normalize_windows(X_train_raw)
    X_val = processor.normalize_windows(X_val_raw)
    X_test = processor.normalize_windows(X_test_raw)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    print(f"Classes ({len(le.classes_)}): {le.classes_}")
    return X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, le, processor.scaler


# ==================== VISUALIZATION ====================
def plot_confusion_matrices(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    tick_marks = np.arange(len(classes))

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix — Test Set (Counts)', fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=11)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_test_counts_9classes.png', dpi=150)
    plt.show()

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=1)
    plt.title('Confusion Matrix — Test Set (Normalized %)', fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j] * 100:.1f}%",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black",
                     fontsize=11)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_test_normalized_9classes.png', dpi=150)
    plt.show()


def plot_per_class_metrics(y_true, y_pred, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    n = len(classes)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    ax.set_xlabel('Gesture Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Precision / Recall / F1-Score (Test Set)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('per_class_metrics_9classes.png', dpi=150)
    plt.show()

    metrics = np.array([precision, recall, f1])
    fig, ax = plt.subplots(figsize=(11, 3.5))
    im = ax.imshow(metrics, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(['Precision', 'Recall', 'F1-Score'])
    ax.set_title('Per-Class Metrics Heatmap (Test Set)', fontsize=13, fontweight='bold')
    for i in range(3):
        for j in range(n):
            val = metrics[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black" if val > 0.5 else "white",
                    fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('per_class_metrics_heatmap_9classes.png', dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis('off')
    table_data = [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
    for i, cls in enumerate(classes):
        table_data.append([cls, f"{precision[i]:.3f}", f"{recall[i]:.3f}",
                           f"{f1[i]:.3f}", str(int(support[i]))])
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()
    table_data.append(['MACRO AVG', f"{macro_p:.3f}", f"{macro_r:.3f}",
                       f"{macro_f1:.3f}", str(int(support.sum()))])
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for j in range(5):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
        table[(len(classes) + 1, j)].set_facecolor('#f39c12')
        table[(len(classes) + 1, j)].set_text_props(fontweight='bold')
    plt.title('Per-Class Metrics Summary (Test Set)',
              fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('per_class_metrics_table_9classes.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n📊 Macro averages → Precision: {macro_p:.3f} | "
          f"Recall: {macro_r:.3f} | F1: {macro_f1:.3f}")


# ==================== MIXUP ====================
def mixup_batch(x, y, alpha=0.15):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


# ==================== TRAINING ====================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test, le, scaler = prepare_data()

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=Config.BATCH_SIZE, shuffle=False)

    model = HandGestureModel(
        input_features=Config.FEATURES,
        num_classes=Config.CLASSES,
        gru_hidden_size=Config.GRU_HIDDEN_SIZE
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ⭐ Focal Loss
    if Config.USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        print(f"🎯 Using Focal Loss (α={Config.FOCAL_ALPHA}, γ={Config.FOCAL_GAMMA})")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # ⭐ Warmup + Cosine
    def lr_lambda(epoch):
        if epoch < Config.WARMUP_EPOCHS:
            return (epoch + 1) / Config.WARMUP_EPOCHS
        return 1.0
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.COSINE_T0,
        T_mult=Config.COSINE_T_MULT,
        eta_min=Config.COSINE_ETA_MIN
    )

    best_val_acc = 0
    best_state = None
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    lr_history = []

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for bx, by in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}"):
            bx, by = bx.to(device), by.to(device)

            mx, ya, yb, lam = mixup_batch(bx, by, Config.MIXUP_ALPHA)

            optimizer.zero_grad()
            out = model(mx)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                _, pred = torch.max(out, 1)
            train_total += by.size(0)
            train_correct += (pred == ya).sum().item()

        train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item()
                _, pred = torch.max(out, 1)
                val_total += by.size(0)
                val_correct += (pred == by).sum().item()
        val_acc = 100 * val_correct / val_total

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # ⭐ Scheduler step
        if epoch < Config.WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        print(f"Epoch {epoch + 1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}% "
              f"| Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f} "
              f"| LR={current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
            print(f"🔥 New best val acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"⏹️ Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ===== TEST =====
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            _, pred = torch.max(out, 1)
            test_correct += (pred == by).sum().item()
            test_total += by.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(by.cpu().numpy())
    test_acc = 100 * test_correct / test_total
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    print(f"🏆 Best Val Acc: {best_val_acc:.2f}%")

    y_true = le.inverse_transform(all_true)
    y_pred = le.inverse_transform(all_preds)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))

    print("\n🎨 Generating visualizations...")
    plot_confusion_matrices(y_true, y_pred, list(le.classes_))
    plot_per_class_metrics(y_true, y_pred, list(le.classes_))

    # ===== SAVE =====
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder_classes': le.classes_,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'config': {
            'window_size': Config.WINDOW_SIZE,
            'window_stride': Config.WINDOW_STRIDE,
            'features': Config.FEATURES,
            'gru_hidden_size': Config.GRU_HIDDEN_SIZE,
            'use_frame_sampling': Config.USE_FRAME_SAMPLING,
            'sample_rate': Config.SAMPLE_RATE,
            'smoothing_window': Config.SMOOTHING_WINDOW,
            'attention_heads': Config.ATTENTION_HEADS,
            'kan_width': Config.KAN_WIDTH,
            'kan_grid': Config.KAN_GRID_SIZE
        }
    }, Config.MODEL_SAVE_PATH)
    print(f"💾 Model saved to {Config.MODEL_SAVE_PATH}")

    # ===== CURVES =====
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.grid(alpha=0.3); plt.title('Loss Curves')
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend(); plt.grid(alpha=0.3); plt.title('Accuracy Curves')
    plt.subplot(1, 3, 3)
    plt.plot(lr_history, label='Learning Rate', color='purple')
    plt.legend(); plt.grid(alpha=0.3); plt.title('LR Schedule')
    plt.tight_layout()
    plt.savefig('training_curves_gru_kan_9classes.png', dpi=150)
    plt.show()

    return model, le, scaler


# ==================== REAL-TIME ====================
def simulate_real_time(file_path, model, label_encoder, scaler, device='cpu'):
    required = Config.REQUIRED_FEATURES
    df = pd.read_csv(file_path)
    for f in required:
        if f not in df.columns:
            df[f] = 0.0
    data = df[required].values.astype(np.float32)
    if Config.USE_FRAME_SAMPLING:
        data = data[::Config.SAMPLE_RATE]

    buffer = []
    pred_history = deque(maxlen=Config.SMOOTHING_WINDOW)
    model.eval()
    print(f"\n🎮 Real-Time Simulation: {os.path.basename(file_path)}")
    print(f"Supported gestures: {list(label_encoder.classes_)}")
    for i in range(len(data)):
        buffer.append(data[i])
        if len(buffer) == Config.WINDOW_SIZE:
            window = np.array(buffer)
            flat = window.reshape(-1, window.shape[-1])
            norm = scaler.transform(flat).reshape(window.shape)
            inp = torch.FloatTensor(norm).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)
                prob = torch.softmax(out, dim=1)
                pred_idx = torch.argmax(out, dim=1).item()
                conf = torch.max(prob).item()
            gesture = label_encoder.inverse_transform([pred_idx])[0]
            pred_history.append(gesture)
            smoothed = Counter(pred_history).most_common(1)[0][0]
            if (i + 1) % 10 == 0:
                print(f"Frame {i + 1}: Raw={gesture} ({conf:.2f}) → Smoothed={smoothed}")
            buffer.pop(0)
    final = Counter(pred_history).most_common(1)[0][0] if pred_history else None
    print(f"\n🏁 Final smoothed prediction: {final}")


# ==================== MAIN ====================
def main():
    print("🤖 Hand Gesture Recognition - FAST + SMART VERSION")
    print("   [Focal Loss + Attention Pooling + Warmup + Residual]")
    print("1. Train new model")
    print("2. Test single file")
    choice = input("Choice (1/2): ").strip()
    if choice == '1':
        train_model()
    elif choice == '2':
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print("No trained model found. Train first.")
            return
        chk = torch.load(Config.MODEL_SAVE_PATH, map_location='cpu', weights_only=False)
        num_classes = len(chk['label_encoder_classes'])
        model = HandGestureModel(
            input_features=Config.FEATURES,
            num_classes=num_classes,
            gru_hidden_size=chk['config']['gru_hidden_size']
        )
        model.load_state_dict(chk['model_state_dict'])
        le = LabelEncoder()
        le.classes_ = chk['label_encoder_classes']
        scaler = StandardScaler()
        scaler.mean_ = chk['scaler_mean']
        scaler.scale_ = chk['scaler_scale']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        fpath = input("Enter CSV file path: ").strip()
        simulate_real_time(fpath, model, le, scaler, device)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
