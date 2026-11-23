from collections import Counter
def majority_vote(verdicts):
    if not verdicts:
        raise ValueError("verdicts cannot be empty")
    return Counter(verdicts).most_common(1)[0][0]
