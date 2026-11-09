"""
config.py

All hyperparameters and settings in ONE place.
Change things here — no need to hunt through other files.
"""

# ─── Data ────────────────────────────────────────────────────────
CDM_CSV = "data/cdms.csv"
PROCESSED_DIR = "data/processed"
SEQ_LEN = 14            # timesteps per sample (max conjunction event length is 14)
PC_THRESHOLD = 5e-4     # elevated risk threshold (~25% positive, 1:3 balance)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_SEED = 42

# ─── Model ───────────────────────────────────────────────────────
D_MODEL = 64           # embedding dimension
N_HEADS = 4            # attention/retention heads
N_LAYERS = 3           # number of RetNet/Transformer blocks
D_FF = 256             # FFN hidden dim (Transformer only)
DROPOUT = 0.1          # dropout rate
GAMMAS = [0.90, 0.95, 0.98, 0.99]  # multi-scale retention decay rates (paper Eq. 2)
POOLING = "last"       # temporal pooling: "last" | "mean" | "max"

# ─── Training ────────────────────────────────────────────────────
EPOCHS = 75
BATCH_SIZE = 32
LR = 4e-4              # learning rate
WEIGHT_DECAY = 1e-5    # AdamW weight decay
WARMUP_RATIO = 0.1     # fraction of steps for linear warmup
FOCAL_ALPHA = 0.75     # Focal Loss alpha (balance weight for ~25% pos vs 75% neg)
FOCAL_GAMMA = 2.0      # Focal Loss gamma (focus on hard examples)
PATIENCE = 20          # early stopping patience
MAX_GRAD_NORM = 1.0    # gradient clipping

# ─── Paths ──────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

# ─── Derived ────────────────────────────────────────────────────
INPUT_DIM = None  # set automatically by data_processing.py based on CSV
