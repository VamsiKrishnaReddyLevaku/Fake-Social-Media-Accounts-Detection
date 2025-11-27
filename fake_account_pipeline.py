import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_auc_score, roc_curve
)
import joblib

warnings.filterwarnings("ignore")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def detect_target(df):
    candidates = [c for c in df.columns if c.lower() in (
        "label", "is_fake", "fake", "target", "isbot", "bot", "class"
    )]
    if candidates:
        return candidates[0]

    for c in df.columns:
        if df[c].dropna().nunique() == 2:
            return c
    return None


def basic_inspection(df, outdir):
    info = []
    info.append(f"Shape: {df.shape}")
    info.append("\nDtypes and non-null counts:")
    info.append(str(df.dtypes.to_frame("dtype").join(df.count().to_frame("non_null"))))
    info.append("\nFirst 5 rows:")
    info.append(str(df.head().to_string()))
    miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    info.append("\nMissing % per column:")
    info.append(str(miss[miss > 0].to_string()))

    with open(os.path.join(outdir, "inspection.txt"), "w") as f:
        f.write("\n\n".join(info))

    print("Saved inspection summary.")


def create_basic_features(df):
    df = df.copy()
    date_cols = [c for c in df.columns if any(x in c.lower() for x in ("date", "created", "joined"))]

    for dc in date_cols:
        try:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
        except:
            pass

    for dc in date_cols:
        if pd.api.types.is_datetime64_any_dtype(df[dc]):
            df["account_age_days"] = (pd.Timestamp.now() - df[dc]).dt.days
            break

    if {"followers_count", "friends_count"}.issubset(df.columns):
        df["follower_following_ratio"] = df["followers_count"] / df["friends_count"].replace(0, np.nan)

    if "statuses_count" in df.columns and "account_age_days" in df.columns:
        df["posts_per_day"] = df["statuses_count"] / df["account_age_days"].replace(0, np.nan)

    return df


def cap_outliers(df, numeric_cols):
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower, upper)
    return df


def plot_class_balance(y, outdir):
    ensure_dir(os.path.join(outdir, "plots"))
    plt.figure(figsize=(6, 4))
    y.value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.savefig(os.path.join(outdir, "plots/class_counts.png"))
    plt.close()


def plot_distributions(df, numeric_cols, outdir):
    ensure_dir(os.path.join(outdir, "plots"))
    for col in numeric_cols[:4]:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col].dropna(), bins=40)
        plt.title(f"Distribution: {col}")
        plt.savefig(os.path.join(outdir, f"plots/dist_{col}.png"))
        plt.close()


def plot_roc(results, y_test, outdir):
    plt.figure(figsize=(8, 6))
    for name, info in results.items():
        if info["y_prob_test"] is not None:
            fpr, tpr, _ = roc_curve(y_test, info["y_prob_test"])
            auc = roc_auc_score(y_test, info["y_prob_test"])
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    ensure_dir(os.path.join(outdir, "plots"))
    plt.savefig(os.path.join(outdir, "plots/roc_curve.png"))
    plt.close()


def main(args):
    ensure_dir(args.outdir)

    
    dataset_path = args.input
    if not dataset_path or not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.getcwd(), "fake_dataset.xlsx")
        if os.path.exists(dataset_path):
            print("Auto-detected dataset:", dataset_path)
        else:
            raise FileNotFoundError(
                "Dataset fake_dataset.xlsx NOT found in project folder. "
                "Place it next to this script or pass --input manually."
            )

    print("Loading:", dataset_path)
    df = pd.read_excel(dataset_path) if dataset_path.endswith(".xlsx") else pd.read_csv(dataset_path)

    print("Data loaded:", df.shape)

    
    basic_inspection(df, args.outdir)

    df = create_basic_features(df)

    id_cols = [c for c in df.columns if any(x in c.lower() for x in ("id", "uuid", "handle"))]
    df.drop(columns=id_cols, inplace=True, errors="ignore")

    
    high_missing = df.columns[(df.isnull().mean() > 0.7)]
    df.drop(columns=high_missing, inplace=True)

    
    target = args.target or detect_target(df)
    if not target:
        raise ValueError("Could not detect target variable. Provide --target manually.")
    print("Target:", target)

    y = df[target]
    X = df.drop(columns=[target])

    
    y = y.replace({"fake": 1, "real": 0, "bot": 1, "yes": 1, "no": 0})
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y.fillna("missing"))

    
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    
    X[numeric] = cap_outliers(X[numeric], numeric)

    
    plot_class_balance(y, args.outdir)
    plot_distributions(X, numeric, args.outdir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    
    numeric_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transform, numeric),
        ("cat", categorical_transform, categorical)
    ])

    models = {
        "LogisticRegression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "RandomForest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=args.seed))
        ]),
        "GradientBoosting": Pipeline([
            ("pre", preprocessor),
            ("clf", GradientBoostingClassifier(n_estimators=150, random_state=args.seed))
        ])
    }

    results = {}
    for name, model in models.items():
        print("Training:", name)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        try:
            y_prob_test = model.predict_proba(X_test)[:, 1]
        except:
            y_prob_test = None

        results[name] = {
            "test_acc": accuracy_score(y_test, y_pred_test),
            "train_acc": accuracy_score(y_train, y_pred_train),
            "y_pred_test": y_pred_test,
            "y_prob_test": y_prob_test
        }

        print(f"{name} → Train: {results[name]['train_acc']:.4f}, Test: {results[name]['test_acc']:.4f}")

    
    plot_roc(results, y_test, args.outdir)

    
    best = max(results, key=lambda m: results[m]["test_acc"])
    joblib.dump(models[best], os.path.join(args.outdir, "best_model.joblib"))
    print("Saved best model:", best)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="fake_dataset.xlsx")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="./outputs")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

