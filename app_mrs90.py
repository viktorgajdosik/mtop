import json
import pathlib
import pickle
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier  # for type hints
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

# Extra imports for debug
import platform
import hashlib
import lightgbm

# Try to import SHAP
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ---------------------------------------------------------
# Paths – adjust if your catalog uses different locations
# ---------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "06_models"
    / "modeling_mrs90"
    / "mrs90_lgbm_model.pkl"
)

X_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "06_models"
    / "modeling_mrs90"
    / "mrs90_X_train.parquet"
)

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "05_model_input"
    / "mt_lightgbm_ready.parquet"
)

CALIBRATION_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "08_reporting"
    / "modeling_mrs90"
    / "calibration"
    / "mrs90_calibration_metrics_raw.json"
)

CV_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "08_reporting"
    / "modeling_mrs90"
    / "cv"
    / "mrs90_lgbm_cv_metrics_raw_10x20.json"
)

DCA_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "08_reporting"
    / "modeling_mrs90"
    / "decision_curve"
    / "mrs90_decision_curve_raw.csv"
)

APP_PASSWORD_KEY = "app_password"  # key in st.secrets


# ---------------------------------------------------------
# Password protection
# ---------------------------------------------------------
def check_password() -> bool:
    """
    Simple password gate using Streamlit secrets.

    - If st.secrets[APP_PASSWORD_KEY] is set, require that password.
    - If not set, show a warning and allow access (useful for local dev).
    """

    secret_password = None
    try:
        # st.secrets behaves like a dict when secrets are configured
        if APP_PASSWORD_KEY in st.secrets:
            secret_password = st.secrets[APP_PASSWORD_KEY]
    except Exception:
        # st.secrets may not exist locally; ignore and run unprotected
        secret_password = None

    # If no password is configured, run app without protection
    if secret_password is None:
        st.warning(
            "App password is not configured in Streamlit secrets. "
            "Running without authentication. "
            "Set `app_password` in Streamlit Cloud secrets to protect this app."
        )
        return True

    # If password already validated in this session, let user through
    if st.session_state.get("password_correct", False):
        return True

    def _password_entered():
        """Callback when user submits password."""
        if st.session_state.get("password_input", "") == str(secret_password):
            st.session_state["password_correct"] = True
            # Don't store the actual password
            st.session_state["password_input"] = ""
        else:
            st.session_state["password_correct"] = False

    st.markdown("### Restricted access")
    st.write("Please enter the access password to use this prognostic tool.")

    st.text_input(
        "Password",
        type="password",
        key="password_input",
        on_change=_password_entered,
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password. Please try again.")

    return False


# ---------------------------------------------------------
# Load model + training data
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model() -> LGBMClassifier:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data(show_spinner=True)
def load_training_data() -> pd.DataFrame:
    return pd.read_parquet(X_TRAIN_PATH)


@st.cache_data(show_spinner=True)
def load_raw_data() -> Optional[pd.DataFrame]:
    """Raw dataset with real categorical labels (mt_lightgbm_ready)."""
    try:
        return pd.read_parquet(RAW_DATA_PATH)
    except FileNotFoundError:
        st.error(f"Raw data file not found at: {RAW_DATA_PATH}")
        return None
    except Exception as e:
        st.error(f"Error loading RAW_DATA_PATH ({RAW_DATA_PATH}): {e}")
        return None


@st.cache_data(show_spinner=True)
def load_calibration_metrics() -> Dict:
    """Load calibration / test metrics JSON, if available."""
    try:
        with open(CALIBRATION_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=True)
def load_cv_metrics() -> Dict:
    """Load cross-validation metrics JSON, if available."""
    try:
        with open(CV_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=True)
def load_dca() -> Optional[pd.DataFrame]:
    """Load decision-curve analysis CSV, if available."""
    try:
        return pd.read_csv(DCA_CSV_PATH)
    except FileNotFoundError:
        return None
    except Exception:
        return None


@st.cache_resource(show_spinner=True)
def get_shap_explainer(_model: LGBMClassifier):
    """
    Cached SHAP TreeExplainer.
    Using `_model` prevents Streamlit from trying to hash the model object.
    """
    if not HAS_SHAP:
        return None

    try:
        return shap.TreeExplainer(_model)
    except Exception as e:
        # Still allow the app to run if SHAP has compatibility issues
        st.warning(
            "SHAP explainer could not be created. "
            "This may happen with some LightGBM versions.\n\n"
            f"Error: {e}"
        )
        return None


def dca_useful_range(df_dca: Optional[pd.DataFrame]) -> Tuple[Optional[float], Optional[float]]:
    """
    From the decision-curve table, find the threshold range where
    the model has higher net benefit than both 'treat all' and 'treat none',
    and has positive net benefit.
    """
    if df_dca is None or df_dca.empty:
        return None, None

    df = df_dca.copy()
    required_cols = {"threshold", "net_benefit_model", "net_benefit_all", "net_benefit_none"}
    if not required_cols.issubset(df.columns):
        return None, None

    mask = (
        (df["net_benefit_model"] > df["net_benefit_all"])
        & (df["net_benefit_model"] > df["net_benefit_none"])
        & (df["net_benefit_model"] > 0)
    )
    df_good = df[mask]
    if df_good.empty:
        return None, None

    t_low = float(df_good["threshold"].min())
    t_high = float(df_good["threshold"].max())
    return t_low, t_high


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def is_binary_numeric(series: pd.Series) -> bool:
    """
    Detect if a column is numeric and essentially 0/1.
    Works with nullable Int64, floats, etc.
    """
    if not is_numeric_dtype(series):
        return False
    vals = series.dropna().unique()
    if len(vals) == 0:
        return False
    try:
        vals_float = np.unique(vals.astype(float))
    except Exception:
        return False
    return set(vals_float).issubset({0.0, 1.0})


def build_patient_row(
    X_train: pd.DataFrame,
    manual_values: dict,
) -> pd.DataFrame:
    """
    Create a 1-row DataFrame with all model features in the correct order.

    - For features the user didn't set manually, fill with train median (numeric)
      or mode (non-numeric).
    - If the user explicitly chose "Missing", the value will be np.nan and NOT
      overwritten here.
    - After constructing the row, *_missing indicator features are forced to be:
         base_missing = 1 if base feature is NaN else 0
      to match the modeling logic.
    """
    feature_order = list(X_train.columns)

    median_vals = {}
    mode_vals = {}
    for col in feature_order:
        s = X_train[col]
        if is_numeric_dtype(s):
            median_vals[col] = float(s.median())
        else:
            if s.dropna().empty:
                mode_vals[col] = np.nan
            else:
                mode_vals[col] = s.dropna().mode().iloc[0]

    data = {}
    for feat in feature_order:
        if feat in manual_values:
            data[feat] = manual_values[feat]
        else:
            if feat in median_vals:
                data[feat] = median_vals[feat]
            else:
                data[feat] = mode_vals.get(feat, np.nan)

    df_row = pd.DataFrame([data], columns=feature_order)

    # ---- CRITICAL: force dtypes to match X_train (LightGBM categoricals) ----
    for col in feature_order:
        train_col = X_train[col]
        train_dtype = train_col.dtype
        try:
            if is_categorical_dtype(train_dtype):
                # Use same categories as training data
                df_row[col] = pd.Categorical(
                    df_row[col],
                    categories=train_col.cat.categories,
                )
            else:
                df_row[col] = df_row[col].astype(train_dtype)
        except Exception:
            # If conversion fails, leave as-is (better than crashing)
            pass

    # Enforce *_missing indicators based on NaN of base vars
    for col in feature_order:
        if col.endswith("_missing"):
            base = col[:-8]  # remove "_missing"
            if base in df_row.columns:
                df_row[col] = df_row[base].isna().astype(int)

    return df_row


def make_shap_safe_row(row: pd.DataFrame, X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Create a version of the row with no NaN values so SHAP does not break.
    This is ONLY for SHAP; predictions still use the original row.
    """
    safe = row.copy()
    for col in safe.columns:
        if pd.isna(safe.at[0, col]):
            col_series = X_train[col]
            if is_numeric_dtype(col_series):
                safe.at[0, col] = float(col_series.median())
            else:
                if col_series.dropna().empty:
                    safe.at[0, col] = ""
                else:
                    safe.at[0, col] = col_series.dropna().mode().iloc[0]
    return safe


def build_category_mapping(
    X_train: pd.DataFrame,
    raw_df: Optional[pd.DataFrame],
    columns: List[str],
) -> Dict[str, Dict[str, object]]:
    """
    For each column, infer mapping from raw labels to encoded values by
    aligning mt_lightgbm_ready (raw_df) with X_train.

    Returns: {col_name: {raw_label: encoded_value, ...}, ...}
    """
    mapping: Dict[str, Dict[str, object]] = {}
    if raw_df is None:
        return mapping

    # First try index alignment (Kedro style)
    common_idx = X_train.index.intersection(raw_df.index)
    if not common_idx.empty:
        Xc = X_train.loc[common_idx]
        Rc = raw_df.loc[common_idx]

        for col in columns:
            if col not in Xc.columns or col not in Rc.columns:
                continue
            df_pairs = pd.DataFrame({"raw": Rc[col], "enc": Xc[col]}).dropna()
            if df_pairs.empty:
                continue
            grouped = df_pairs.groupby("raw")["enc"].agg(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
            )
            mapping[col] = grouped.to_dict()

        return mapping

    # If no overlapping index but we have patient_id in both, align via merge
    if "patient_id" in X_train.columns and "patient_id" in raw_df.columns:
        # Restrict to columns + patient_id that actually exist in each frame
        cols_raw = ["patient_id"] + [c for c in columns if c in raw_df.columns]
        cols_enc = ["patient_id"] + [c for c in columns if c in X_train.columns]

        merged = (
            raw_df[cols_raw]
            .merge(
                X_train[cols_enc],
                on="patient_id",
                suffixes=("_raw", "_enc"),
            )
        )

        for col in columns:
            raw_col = f"{col}_raw"
            enc_col = f"{col}_enc"
            if raw_col not in merged.columns or enc_col not in merged.columns:
                continue

            df_pairs = merged[[raw_col, enc_col]].dropna()
            if df_pairs.empty:
                continue

            grouped = df_pairs.groupby(raw_col)[enc_col].agg(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
            )
            mapping[col] = grouped.to_dict()

        return mapping

    # If we get here, we couldn't align raw_df and X_train
    return mapping


# ---------------------------------------------------------
# Debug helpers (file hashes)
# ---------------------------------------------------------
def _file_md5(path: pathlib.Path) -> str:
    try:
        with open(path, "rb") as f:
            h = hashlib.md5()
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "MISSING"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------
# App body (runs only after password check)
# ---------------------------------------------------------
def run_app():
    st.title("mRS90 outcome – LightGBM prognostic model")
    st.caption("Predicted probability of good outcome (mRS 0–2 at 90 days).")

    # Load stuff
    model = load_model()
    X_train = load_training_data()
    raw_df = load_raw_data()
    calib_metrics = load_calibration_metrics()
    cv_metrics = load_cv_metrics()
    dca_df = load_dca()
    dca_low, dca_high = dca_useful_range(dca_df)

    # Multicategorical variables to map from raw labels
    MULTICAT_VARS = [
        "thrombolytics",
        "occlusion_site",
        "hemisphere",
        "ivt_different_hospital",
        "transfer_from_other_hospital",
        "antithrombotics_before",
        "heart_condition",
        "etiology",
    ]
    MULTICAT_VARS = [c for c in MULTICAT_VARS if c in X_train.columns]

    cat_mapping = build_category_mapping(X_train, raw_df, MULTICAT_VARS)

    # Constants for posterior vs anterior consistency
    POSTERIOR_OCCLUSION_LABELS = {"BA", "AB+AV", "VA", "PCA"}
    POSTERIOR_HEMISPHERE_LABEL = "Posterior circulation"
    LEFT_HEMISPHERE_LABEL = "Left hemisphere"
    RIGHT_HEMISPHERE_LABEL = "Right hemisphere"

    # Sidebar – about
    st.sidebar.header("About")
    st.sidebar.write(
        """
This app uses a pre-procedural LightGBM model predicting the
probability of good functional outcome (mRS 0–2 at 90 days)
after mechanical thrombectomy.

You can set key predictors for a single patient; all other
model features are fixed to median (or most frequent) values
from the training set.

⚠️ Research use only – not clinical decision support.
"""
    )

    # Debug mode (for encoded vector + SHAP + env info)
    debug_mode = st.sidebar.checkbox(
        "Debug mode (env, artifacts, encoded vector / SHAP)",
        value=False,
    )

    # ---------------------------
    # Debug: environment + artifacts + cache controls
    # ---------------------------
    if debug_mode:
        st.markdown("### Debug: environment & artifacts")

        st.json(
            {
                "python": platform.python_version(),
                "lightgbm": lightgbm.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            }
        )

        st.markdown("#### Debug: model & data file hashes")
        st.json(
            {
                "MODEL_PATH": str(MODEL_PATH),
                "MODEL_MD5": _file_md5(MODEL_PATH),
                "X_TRAIN_PATH": str(X_TRAIN_PATH),
                "X_TRAIN_MD5": _file_md5(X_TRAIN_PATH),
                "RAW_DATA_PATH": str(RAW_DATA_PATH),
                "RAW_DATA_MD5": _file_md5(RAW_DATA_PATH),
                "CALIBRATION_METRICS_PATH": str(CALIBRATION_METRICS_PATH),
                "CALIBRATION_METRICS_MD5": _file_md5(CALIBRATION_METRICS_PATH),
                "CV_METRICS_PATH": str(CV_METRICS_PATH),
                "CV_METRICS_MD5": _file_md5(CV_METRICS_PATH),
                "DCA_CSV_PATH": str(DCA_CSV_PATH),
                "DCA_CSV_MD5": _file_md5(DCA_CSV_PATH),
            }
        )

        st.markdown("#### Debug: Streamlit cache controls")
        if st.button("Clear all Streamlit caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared. Please rerun the app.")

    st.markdown("### Patient characteristics")

    # Sectioned layout
    sections = [
        (
            "Demographics",
            ["age", "sex", "bmi"],
        ),
        (
            "Baseline clinical",
            [
                "hypertension",
                "diabetes",
                "hyperlipidemia",
                "smoking",
                "alcohol_abuse",
                "arrhythmia",
                "tia_before",
                "cmp_before",
                "heart_condition",
                "antithrombotics_before",
                "statins_before",
                "systolic_bp",
                "diastolic_bp",
                "glycemia",
                "cholesterol",
                "mrs_before",
                "admission_nihss",
            ],
        ),
        (
            "Imaging",
            [
                "aspects",
                "occlusion_site",
                "hemisphere",
            ],
        ),
        (
            "Treatment / timing",
            [
                "ivt_given",
                "thrombolytics",
                "ivt_different_hospital",
                "transfer_from_other_hospital",
                "onset_to_ivt_min",
                "onset_to_puncture_min",
                "etiology",
            ],
        ),
    ]

    manual_values: Dict[str, object] = {}

    # IVT state shared across sections
    ivt_given_flag: Optional[bool] = None
    ivt_missing_flag: bool = False

    # ---------------------------
    # Input UI by sections
    # ---------------------------
    for section_title, feat_list in sections:
        feat_list = [f for f in feat_list if f in X_train.columns]
        if not feat_list:
            continue

        st.markdown(f"#### {section_title}")
        cols = st.columns(2)
        col_idx = 0

        for feat in feat_list:
            col_series = X_train[feat]
            name_lower = feat.lower()

            # ---------------------------------------------
            # Special handling: SEX (with "Missing" option)
            # ---------------------------------------------
            if feat == "sex":
                s = col_series.dropna()
                unique_vals = s.unique()

                if is_numeric_dtype(s) and len(unique_vals) > 0:
                    try:
                        codes = np.unique(unique_vals.astype(float))
                    except Exception:
                        codes = unique_vals

                    if len(codes) == 2:
                        code_female = float(min(codes))
                        code_male = float(max(codes))
                    elif len(codes) == 1:
                        code_female = code_male = float(codes[0])
                    else:
                        code_female = 0.0
                        code_male = 1.0

                    if not s.mode().empty:
                        mode_code = float(s.mode().iloc[0])
                        default_label = "Female" if mode_code == code_female else "Male"
                    else:
                        default_label = "Female"

                    label_to_val = {
                        "Missing": np.nan,
                        "Female": code_female,
                        "Male": code_male,
                    }
                    options = list(label_to_val.keys())
                    default_idx = options.index(default_label)

                    with cols[col_idx]:
                        selected = st.selectbox(
                            "Sex",
                            options=options,
                            index=default_idx,
                            help="Mapped internally to the numeric encoding used in training data.",
                        )
                    manual_values["sex"] = label_to_val[selected]
                    col_idx = 1 - col_idx
                    continue

                # Non-numeric sex
                non_null = col_series.dropna().unique()
                raw_options = list(non_null)
                label_to_val = {"Missing": np.nan}
                for v in raw_options:
                    label_to_val[str(v)] = v

                options = list(label_to_val.keys())
                default_idx = 0

                with cols[col_idx]:
                    selected = st.selectbox(
                        "Sex",
                        options=options,
                        index=default_idx,
                        help="Options taken directly from training data, plus 'Missing'.",
                    )
                manual_values["sex"] = label_to_val[selected]
                col_idx = 1 - col_idx
                continue

            # -------------------------------------------------
            # Special handling: IVT_GIVEN (checkbox + Missing)
            # -------------------------------------------------
            if feat == "ivt_given":
                with cols[col_idx]:
                    ivt_missing_flag = st.checkbox(
                        "IVT information missing",
                        value=False,
                        key="ivt_missing",
                    )
                    if ivt_missing_flag:
                        ivt_given_flag = False
                        manual_values["ivt_given"] = np.nan
                        st.checkbox(
                            "IVT given",
                            value=False,
                            disabled=True,
                            key="ivt_given_disabled",
                        )
                    else:
                        s = col_series.dropna()
                        if not s.empty:
                            default_val = int(s.mode().iloc[0])
                            default_bool = bool(default_val)
                        else:
                            default_bool = False

                        ivt_given_flag = st.checkbox(
                            "IVT given",
                            value=default_bool,
                            key="ivt_given",
                        )
                        manual_values["ivt_given"] = 1 if ivt_given_flag else 0

                col_idx = 1 - col_idx
                continue

            # -------------------------------------------------
            # Gating: ONSET_TO_IVT_MIN
            # -------------------------------------------------
            if feat == "onset_to_ivt_min":
                with cols[col_idx]:
                    if not ivt_given_flag:
                        st.text_input(
                            "Onset-to-IVT (min)",
                            value="Not applicable (no IVT / missing)",
                            disabled=True,
                        )
                        manual_values["onset_to_ivt_min"] = np.nan
                    else:
                        s = col_series
                        lo = float(s.quantile(0.01))
                        hi = float(s.quantile(0.99))
                        default = float(s.median())
                        if lo == hi:
                            lo = float(s.min())
                            hi = float(s.max())
                        if lo == hi:
                            lo -= 1.0
                            hi += 1.0

                        missing = st.checkbox(
                            "Onset-to-IVT missing",
                            value=False,
                            key="onset_to_ivt_missing",
                        )
                        if missing:
                            manual_values["onset_to_ivt_min"] = np.nan
                            st.slider(
                                label="Onset-to-IVT (min)",
                                min_value=float(np.floor(lo)),
                                max_value=float(np.ceil(hi)),
                                value=float(default),
                                step=1.0,
                                disabled=True,
                            )
                        else:
                            val = st.slider(
                                label="Onset-to-IVT (min)",
                                min_value=float(np.floor(lo)),
                                max_value=float(np.ceil(hi)),
                                value=float(default),
                                step=1.0,
                                help=f"Training 1–99th percentile: [{lo:.1f}, {hi:.1f}]",
                            )
                            manual_values["onset_to_ivt_min"] = float(val)

                col_idx = 1 - col_idx
                continue

            # -------------------------------------------------
            # Special handling: OCCLUSION_SITE with hemisphere gating
            # -------------------------------------------------
            if feat == "occlusion_site":
                col_map = cat_mapping.get("occlusion_site", {})
                if col_map:
                    raw_labels = sorted(col_map.keys(), key=lambda x: str(x))
                else:
                    raw_labels = sorted(col_series.dropna().unique(), key=lambda x: str(x))
                    col_map = {str(v): v for v in raw_labels}

                # Get previously selected hemisphere label (string from selectbox)
                hemi_label = st.session_state.get("hemisphere_sel", None)

                allowed_raw_labels = raw_labels
                if hemi_label == POSTERIOR_HEMISPHERE_LABEL:
                    # Posterior hemisphere -> only BA / AB+AV / VA / PCA
                    allowed_raw_labels = [
                        lbl for lbl in raw_labels if lbl in POSTERIOR_OCCLUSION_LABELS
                    ]
                elif hemi_label in (LEFT_HEMISPHERE_LABEL, RIGHT_HEMISPHERE_LABEL):
                    # Left/Right hemisphere -> exclude posterior occlusion sites
                    allowed_raw_labels = [
                        lbl for lbl in raw_labels if lbl not in POSTERIOR_OCCLUSION_LABELS
                    ]

                label_to_val = {"Missing": np.nan}
                for lbl in allowed_raw_labels:
                    enc = col_map.get(lbl, lbl)
                    label_to_val[str(lbl)] = enc
                options = list(label_to_val.keys())

                # Use previous selection if still valid
                prev = st.session_state.get("occlusion_site_sel", None)
                default_label = prev if prev in options else "Missing"
                default_idx = options.index(default_label)

                with cols[col_idx]:
                    selected = st.selectbox(
                        "occlusion_site",
                        options=options,
                        index=default_idx,
                        help="Categories taken from mt_lightgbm_ready (raw labels), plus 'Missing'.",
                        key="occlusion_site_sel",
                    )
                manual_values["occlusion_site"] = label_to_val[selected]
                col_idx = 1 - col_idx
                continue

            # -------------------------------------------------
            # Special handling: HEMISPHERE with occlusion gating
            # -------------------------------------------------
            if feat == "hemisphere":
                col_map = cat_mapping.get("hemisphere", {})
                if col_map:
                    raw_labels = sorted(col_map.keys(), key=lambda x: str(x))
                else:
                    raw_labels = sorted(col_series.dropna().unique(), key=lambda x: str(x))
                    col_map = {str(v): v for v in raw_labels}

                occ_label = st.session_state.get("occlusion_site_sel", None)

                allowed_raw_labels = raw_labels
                if occ_label in POSTERIOR_OCCLUSION_LABELS:
                    # Posterior occlusion -> only posterior hemisphere
                    allowed_raw_labels = [
                        lbl for lbl in raw_labels if lbl == POSTERIOR_HEMISPHERE_LABEL
                    ]
                elif occ_label is not None and occ_label not in ("Missing",) and occ_label not in POSTERIOR_OCCLUSION_LABELS:
                    # Non-posterior occlusion -> forbid posterior hemisphere
                    allowed_raw_labels = [
                        lbl for lbl in raw_labels if lbl != POSTERIOR_HEMISPHERE_LABEL
                    ]

                label_to_val = {"Missing": np.nan}
                for lbl in allowed_raw_labels:
                    enc = col_map.get(lbl, lbl)
                    label_to_val[str(lbl)] = enc
                options = list(label_to_val.keys())

                prev = st.session_state.get("hemisphere_sel", None)
                default_label = prev if prev in options else "Missing"
                default_idx = options.index(default_label)

                with cols[col_idx]:
                    selected = st.selectbox(
                        "hemisphere",
                        options=options,
                        index=default_idx,
                        help="Categories taken from mt_lightgbm_ready (raw labels), plus 'Missing'.",
                        key="hemisphere_sel",
                    )
                manual_values["hemisphere"] = label_to_val[selected]
                col_idx = 1 - col_idx
                continue

            # -------------------------------------------------
            # THROMBOLYTICS / IVT_DIFFERENT_HOSPITAL (multicategory, gated)
            # -------------------------------------------------
            if feat in ("thrombolytics", "ivt_different_hospital"):
                col_map = cat_mapping.get(feat, {})
                if col_map:
                    raw_labels = sorted(col_map.keys(), key=lambda x: str(x))
                else:
                    raw_labels = sorted(col_series.dropna().unique(), key=lambda x: str(x))
                    col_map = {str(v): v for v in raw_labels}

                label_to_val = {"Missing": np.nan}
                for lbl, enc in col_map.items():
                    label_to_val[str(lbl)] = enc
                options = list(label_to_val.keys())

                label_text = "Thrombolytics" if feat == "thrombolytics" else "IVT given at different hospital"

                with cols[col_idx]:
                    if not ivt_given_flag:
                        st.selectbox(
                            label_text,
                            options=options,
                            index=0,
                            disabled=True,
                            help="Only applicable if IVT was given.",
                            key=f"{feat}_disabled",
                        )
                        manual_values[feat] = np.nan
                    else:
                        selected = st.selectbox(
                            label_text,
                            options=options,
                            index=0,
                            help="Options taken from mt_lightgbm_ready (raw labels), plus 'Missing'.",
                            key=f"{feat}_sel",
                        )
                        manual_values[feat] = label_to_val[selected]

                col_idx = 1 - col_idx
                continue

            # -------------------------------------------------
            # Generic MULTICAT variables (not gated by IVT or posterior rules)
            # -------------------------------------------------
            if feat in MULTICAT_VARS and feat not in ("occlusion_site", "hemisphere"):
                col_map = cat_mapping.get(feat, {})
                if col_map:
                    raw_labels = sorted(col_map.keys(), key=lambda x: str(x))
                else:
                    raw_labels = sorted(col_series.dropna().unique(), key=lambda x: str(x))
                    col_map = {str(v): v for v in raw_labels}

                label_to_val = {"Missing": np.nan}
                for lbl, enc in col_map.items():
                    label_to_val[str(lbl)] = enc
                options = list(label_to_val.keys())

                with cols[col_idx]:
                    selected = st.selectbox(
                        feat,
                        options=options,
                        index=0,
                        help="Categories taken from mt_lightgbm_ready (raw labels), plus 'Missing'.",
                        key=f"{feat}_multicat",
                    )
                manual_values[feat] = label_to_val[selected]
                col_idx = 1 - col_idx
                continue

            # -------------------------------------------------
            # BINARY NUMERIC -> tri-state selectbox (Missing/No/Yes)
            # -------------------------------------------------
            if is_binary_numeric(col_series):
                s = col_series.dropna()
                if not s.empty:
                    mode_val = int(s.mode().iloc[0])
                else:
                    mode_val = 0

                label_to_val = {
                    "Missing": np.nan,
                    "No (0)": 0,
                    "Yes (1)": 1,
                }
                default_label = "Yes (1)" if mode_val == 1 else "No (0)"
                options = list(label_to_val.keys())
                default_idx = options.index(default_label)

                with cols[col_idx]:
                    selected = st.selectbox(
                        feat,
                        options=options,
                        index=default_idx,
                        help="Binary variable encoded as 0/1 in training data, plus 'Missing'.",
                        key=f"{feat}_bin",
                    )
                manual_values[feat] = label_to_val[selected]
                col_idx = 1 - col_idx
                continue

            # -------------------------------------------------
            # NUMERIC non-binary -> slider + "Missing" checkbox
            # -------------------------------------------------
            if is_numeric_dtype(col_series):
                s = col_series
                lo = float(s.quantile(0.01))
                hi = float(s.quantile(0.99))
                default = float(s.median())

                if lo == hi:
                    lo = float(s.min())
                    hi = float(s.max())
                if lo == hi:
                    lo -= 1.0
                    hi += 1.0

                step = 1.0
                if any(k in name_lower for k in ["glyc", "chol", "bmi", "bp"]):
                    step = 0.1
                if "min" in name_lower:
                    step = 1.0

                with cols[col_idx]:
                    missing = st.checkbox(
                        f"{feat} missing",
                        value=False,
                        key=f"{feat}_missing",
                    )
                    if missing:
                        manual_values[feat] = np.nan
                        st.slider(
                            label=f"{feat}",
                            min_value=float(np.floor(lo)),
                            max_value=float(np.ceil(hi)),
                            value=float(default),
                            step=step,
                            disabled=True,
                            key=f"{feat}_slider_disabled",
                        )
                    else:
                        val = st.slider(
                            label=f"{feat}",
                            min_value=float(np.floor(lo)),
                            max_value=float(np.ceil(hi)),
                            value=float(default),
                            step=step,
                            help=f"Training 1–99th percentile: [{lo:.1f}, {hi:.1f}]",
                            key=f"{feat}_slider",
                        )
                        manual_values[feat] = float(val)

                col_idx = 1 - col_idx
                continue

            # -------------------------------------------------
            # NON-NUMERIC fallback -> categories + "Missing"
            # -------------------------------------------------
            non_null = col_series.dropna().unique()
            raw_options = list(non_null)
            label_to_val = {"Missing": np.nan}
            for v in raw_options:
                label_to_val[str(v)] = v
            options = list(label_to_val.keys())

            with cols[col_idx]:
                selected = st.selectbox(
                    feat,
                    options=options,
                    index=0,
                    help="Options taken from training data, plus 'Missing'.",
                    key=f"{feat}_cat",
                )
            manual_values[feat] = label_to_val[selected]
            col_idx = 1 - col_idx

    st.markdown("---")

    # ---------------------------------------------------------
    # Prediction + outputs
    # ---------------------------------------------------------
    if st.button("Compute probability of good outcome (mRS 0–2)"):
        patient_row = build_patient_row(X_train, manual_values)

        proba_good = float(model.predict_proba(patient_row)[0, 1])
        proba_bad = 1.0 - proba_good

        st.subheader("Predicted probabilities")
        st.metric(
            "Good outcome (mRS 0–2)",
            f"{proba_good * 100:.1f} %",
        )
        st.metric(
            "Poor outcome (mRS 3–6)",
            f"{proba_bad * 100:.1f} %",
        )

        # -------------------------------------------------
        # Debug: fixed reference patient (for env comparison)
        # -------------------------------------------------
        debug_patient = {
            "age": 70,
            "admission_nihss": 14,
            "mrs_before": 1,
            "aspects": 9,
            "onset_to_puncture_min": 180.0,
        }
        debug_row = build_patient_row(X_train, debug_patient)
        debug_proba_good = float(model.predict_proba(debug_row)[0, 1])

        if debug_mode:
            st.markdown("#### Debug: fixed reference profile")
            st.write("Reference profile (hard-coded, independent of UI):")
            st.json(debug_patient)
            st.write(
                f"Reference profile – probability of good outcome (mRS 0–2): "
                f"**{debug_proba_good:.4f}**"
            )

        # ------------- Debug: encoded vector + SHAP -------------
        if debug_mode:
            st.markdown("#### Debug: encoded feature vector (current patient)")
            st.write(patient_row.T.rename(columns={0: "value"}))

            st.markdown("#### Debug: SHAP explanation (if available)")
            if not HAS_SHAP:
                st.info(
                    "SHAP package is not installed in this environment. "
                    "Install `shap` to see feature contributions."
                )
            else:
                try:
                    explainer = get_shap_explainer(model)
                    if explainer is None:
                        st.info(
                            "SHAP explainer is not available (initialization failed). "
                            "Check LightGBM/SHAP versions if you need this."
                        )
                    else:
                        safe_row = make_shap_safe_row(patient_row, X_train)
                        shap_values = explainer.shap_values(safe_row)
                        expected_value = explainer.expected_value

                        # For LightGBM binary classifier
                        if isinstance(shap_values, list):
                            shap_for_pos = np.array(shap_values[1])[0]
                            if isinstance(expected_value, (list, np.ndarray)):
                                base_value = expected_value[1]
                            else:
                                base_value = expected_value
                        else:
                            shap_for_pos = np.array(shap_values)[0]
                            base_value = expected_value

                        shap_df = pd.DataFrame(
                            {
                                "feature": safe_row.columns,
                                "value": patient_row.iloc[0].values,
                                "shap": shap_for_pos,
                                "abs_shap": np.abs(shap_for_pos),
                            }
                        ).sort_values("abs_shap", ascending=False)

                        st.write("Top SHAP drivers (positive = pushes towards better outcome):")
                        st.dataframe(shap_df.head(20).set_index("feature"))

                        st.bar_chart(shap_df.head(20).set_index("feature")["shap"])
                        st.caption(f"SHAP base value (log-odds for reference patient): {base_value:.3f}")

                except Exception as e:
                    st.warning(
                        "Could not compute SHAP values. "
                        "Make sure the `shap` package is installed and compatible with LightGBM.\n\n"
                        f"Error: {e}"
                    )

    # ------------------------------------------------------------------
    # Model performance for clinicians – dynamic from saved metrics
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Model performance (validation)")

    auc_test = np.nan
    brier_test = np.nan
    logloss_test = np.nan
    cal_slope = np.nan
    cal_intercept = np.nan

    if calib_metrics:
        test_block = calib_metrics.get("test", {})
        if test_block:
            auc_test = float(test_block.get("auc", np.nan))
            brier_test = float(test_block.get("brier", np.nan))
            logloss_test = float(test_block.get("log_loss", np.nan))
            cal_intercept = float(test_block.get("cal_intercept", np.nan))
            cal_slope = float(test_block.get("cal_slope", np.nan))

        auc_test = float(calib_metrics.get("roc_auc_test", auc_test))
        brier_test = float(calib_metrics.get("brier_test", brier_test))
        logloss_test = float(calib_metrics.get("log_loss_test", logloss_test))
        cal_intercept = float(
            calib_metrics.get("calibration_intercept_test", cal_intercept)
        )
        cal_slope = float(
            calib_metrics.get("calibration_slope_test", cal_slope)
        )

    col_perf1, col_perf2 = st.columns(2)

    with col_perf1:
        st.metric(
            "ROC AUC (test set)",
            f"{auc_test:.2f}" if not np.isnan(auc_test) else "N/A",
        )
        st.write(
            "➤ **Discrimination**\n\n"
            + (
                f"- ROC AUC of **{auc_test:.2f}** means that in about "
                f"**{auc_test*100:.0f} out of 100** random pairs of patients "
                "with different outcomes (one good, one poor), "
                "the model assigns a higher probability of good outcome to "
                "the patient who actually had mRS 0–2."
                if not np.isnan(auc_test)
                else "- ROC AUC on the held-out test set is not available in this build."
            )
        )

    with col_perf2:
        st.metric(
            "Brier score (test set)",
            f"{brier_test:.3f}" if not np.isnan(brier_test) else "N/A",
        )
        st.write(
            "➤ **Calibration**\n\n"
            + (
                f"- Brier score summarizes the average squared difference between "
                "predicted probabilities and actual outcomes (lower is better).\n"
                f"- Calibration slope ≈ **{cal_slope:.2f}** "
                "(1.0 = ideal; <1 → predictions too extreme; >1 → too conservative).\n"
                f"- Calibration intercept ≈ **{cal_intercept:.2f}** "
                "(0 = no systematic over- or underestimation)."
                if not np.isnan(brier_test)
                else "- Calibration metrics (Brier score, slope, intercept) are not available."
            )
        )

    # ----- Decision curve analysis -----
    st.markdown("#### Decision curve analysis")

    if dca_df is not None and not dca_df.empty:
        if dca_low is not None and dca_high is not None:
            st.write(
                f"- **Useful threshold range** where the model has higher net benefit "
                f"than both 'treat all' and 'treat none' and net benefit > 0: "
                f"**{dca_low*100:.0f}–{dca_high*100:.0f} %**."
            )
        else:
            st.write(
                "- Decision-curve CSV loaded, but no threshold range with "
                "clear net benefit advantage was detected by the rule used in this app."
            )

        st.write("Net benefit curves on the test set:")
        dca_plot_df = dca_df.set_index("threshold")[
            ["net_benefit_model", "net_benefit_all", "net_benefit_none"]
        ]
        st.line_chart(dca_plot_df)
    else:
        st.write(
            "- Decision-curve analysis file not found or empty – "
            "no DCA result can be displayed."
        )

    # CV summary
    if cv_metrics and "summary" in cv_metrics:
        summary = cv_metrics["summary"]
        auc_info = summary.get("auc", {})
        brier_info = summary.get("brier", {})
        logloss_info = summary.get("log_loss", {})

        auc_mean = auc_info.get("mean", np.nan)
        auc_sd = auc_info.get("std", np.nan)
        brier_mean = brier_info.get("mean", np.nan)
        brier_sd = brier_info.get("std", np.nan)
        logloss_mean = logloss_info.get("mean", np.nan)
        logloss_sd = logloss_info.get("std", np.nan)

        n_splits = cv_metrics.get("n_splits", None)
        n_repeats = cv_metrics.get("n_repeats", None)

        with st.expander("Cross-validation performance (development data)"):
            lines = []
            if n_splits is not None and n_repeats is not None:
                lines.append(
                    f"- **{n_splits}-fold stratified CV, repeated {n_repeats}×** "
                    "on the development cohort."
                )
            if not np.isnan(auc_mean):
                lines.append(
                    f"- Mean ROC AUC across folds: **{auc_mean:.2f}** "
                    f"(SD {auc_sd:.2f})"
                )
            if not np.isnan(brier_mean):
                lines.append(
                    f"- Mean Brier score: **{brier_mean:.3f}** "
                    f"(SD {brier_sd:.3f})"
                )
            if not np.isnan(logloss_mean):
                lines.append(
                    f"- Mean log loss: **{logloss_mean:.3f}** "
                    f"(SD {logloss_sd:.3f})"
                )
            st.markdown("\n".join(lines))

    # Explanation block
    with st.expander("How should I interpret these numbers as a clinician?"):
        if dca_low is not None and dca_high is not None:
            dca_range_text = (
                f"roughly between **{dca_low:.0%} and {dca_high:.0%}** "
                "(threshold probability for mRS 0–2)."
            )
        else:
            dca_range_text = (
                "in the range of clinically relevant threshold probabilities "
                "(precise range not computed in this app build)."
            )

        st.markdown(
            f"""
- **Discrimination (Who is higher risk?)**  
  The model is reasonably good at *ranking* patients by prognosis.  
  AUC ≈ **{auc_test:.2f}** means that if you take two patients at random,  
  one with good outcome (mRS 0–2) and one with poor outcome (mRS 3–6),  
  the model will give a higher predicted probability of good outcome to  
  the truly good-outcome patient in about **{auc_test*100:.0f}%** of cases.

- **Calibration (Are the probabilities believable?)**  
  On the independent test set, predicted probabilities and observed
  frequencies are **approximately aligned**, although the negative
  intercept (≈ {cal_intercept:.2f}) suggests some overestimation
  of the absolute probability of good outcome on average.

- **Clinical usefulness (Decision curve analysis)**  
  In decision-curve analysis on the test set, the model provided higher
  net benefit than "treat all" or "treat none" {dca_range_text}  
  In that range, using the model could help reduce unnecessary treatment
  in patients with very low chance of good outcome while maintaining
  similar numbers of good outcomes overall.

⚠️ These results are based on the original development / validation cohort  
and may not directly transfer to all patient populations or settings.
"""
        )

    st.markdown("---")
    st.caption(
        "Internal research tool – LightGBM model trained on pre-procedural stroke data "
        "with monotonic constraints. This is **not** a clinical decision support system."
    )


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="mRS90 LightGBM Prognostic Calculator",
        layout="centered",
    )

    # Password gate
    if not check_password():
        # If password is wrong or not yet entered, do not render the app body
        return

    # If password is correct or not required, run the main app
    run_app()


if __name__ == "__main__":
    main()
