from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .contracts import LABEL_COLUMN, NON_FEATURE_COLUMNS

LABEL_COL = LABEL_COLUMN
NON_FEATURE_COLS = NON_FEATURE_COLUMNS

def drop_constant_and_redundant(df: pd.DataFrame, corr_threshold: float = 0.98):
    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numeric = df[features].select_dtypes(include="number")
    constants = [c for c in numeric if numeric[c].nunique(dropna=False) <= 1]
    out = df.drop(columns=constants).copy()
    numeric = out[[c for c in features if c not in constants]].select_dtypes(include="number")
    corr = numeric.corr().abs()
    upper = corr.where(pd.DataFrame(
        [[i < j for j in range(len(corr))] for i in range(len(corr))],
        index=corr.index, columns=corr.columns))
    redundant = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
    return out.drop(columns=redundant)

def fit_scaler(train_df: pd.DataFrame, feature_columns: list):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns])
    return scaler

def transform_with_scaler(df: pd.DataFrame, scaler, feature_columns: list):
    out = df.copy()
    out.loc[:, feature_columns] = scaler.transform(out[feature_columns])
    return out

@dataclass(frozen=True)
class AgentFeatureView:
    agent_id: str
    feature_columns: list

def build_agent_feature_views(all_feature_columns: list):
    flow = [c for c in all_feature_columns if any(k in c for k in ("Duration","IAT","Flow","Rate"))]
    packet = [c for c in all_feature_columns if any(k in c for k in ("Length","Size","Bytes","Packet"))]
    flags = [c for c in all_feature_columns if "Flag" in c or "URG" in c or "ACK" in c or "SYN" in c]
    stats = [c for c in all_feature_columns if c not in set(flow + packet + flags)]
    candidates = [
        AgentFeatureView("agent_A", flow),
        AgentFeatureView("agent_B", packet),
        AgentFeatureView("agent_C", flags),
        AgentFeatureView("agent_D", stats),
    ]
    # Remove empty views; actual heterogeneity must still be measured empirically.
    return [v for v in candidates if v.feature_columns]
