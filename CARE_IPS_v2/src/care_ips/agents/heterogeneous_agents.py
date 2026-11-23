from sklearn.ensemble import RandomForestClassifier
from care_ips.agents.base_agent import DetectionAgent, AgentVerdict

class RandomForestAgent(DetectionAgent):
    def __init__(self, agent_id: str, feature_columns: list, n_estimators: int = 200, seed: int = 42):
        if not feature_columns:
            raise ValueError(f"{agent_id} has an empty feature view")
        self.agent_id = agent_id
        self.feature_columns = list(feature_columns)
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        self._fitted = False

    def fit(self, train_df, label_col: str = "Label"):
        self.model.fit(train_df[self.feature_columns], train_df[label_col])
        self._fitted = True
        return self

    def predict(self, event_features):
        if not self._fitted:
            raise RuntimeError(f"{self.agent_id} must be fit before predict")
        X = event_features[self.feature_columns]
        pred = self.model.predict(X)[0]
        confidence = float(self.model.predict_proba(X)[0].max())
        return AgentVerdict(self.agent_id, str(pred), confidence, frozenset())
