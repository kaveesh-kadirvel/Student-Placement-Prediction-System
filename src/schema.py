from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT_DIR / "train.csv"
TEST_PATH = ROOT_DIR / "test.csv"
MODEL_PATH = ROOT_DIR / "placement_model.pkl"
BENCHMARK_PATH = ROOT_DIR / "placed_avg_stats.pkl"
REPORTS_DIR = ROOT_DIR / "reports"

TARGET_COLUMN = "Placement_Status"
POSITIVE_LABEL = "Placed"
NEGATIVE_LABEL = "Not Placed"
LABEL_MAP = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}

ID_COLUMNS = ["Student_ID"]
NUMERIC_COLUMNS = [
    "Age",
    "CGPA",
    "Internships",
    "Projects",
    "Coding_Skills",
    "Communication_Skills",
    "Aptitude_Test_Score",
    "Soft_Skills_Rating",
    "Certifications",
    "Backlogs",
]
CATEGORICAL_COLUMNS = ["Gender", "Degree", "Branch"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

REQUIRED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
