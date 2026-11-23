"""LLM reasoning interface placeholder.

A provider-specific LLM client is intentionally not selected in this package.
Any future implementation must accept only validated EvidenceItem objects and
must enforce evidence-ID references in a structured response schema.
"""
class ReasoningAgentNotConfigured(RuntimeError):
    pass

class ReasoningAgent:
    def __init__(self, client=None):
        self.client = client

    def reason(self, validated_evidence):
        if self.client is None:
            raise ReasoningAgentNotConfigured("Configure an LLM client before calling reason().")
        raise NotImplementedError("Provider-specific constrained generation is not implemented.")
