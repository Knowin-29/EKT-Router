"""One acquisition per selected provider; typed output or an invalid packet."""

from .evidence_contract import EvidencePacket


class PacketProvider:
    def __init__(self, providers, costs=None):
        self.providers, self.costs = dict(providers), dict(costs or {})
        self.last_packets = {}

    def __call__(self, proposal, question, image):
        packets = {}
        for key, channel, action in proposal.action.components():
            provider_key = f"{channel.value}:{action}"
            provider = self.providers.get(provider_key)
            try:
                if provider is None:
                    raise ValueError("provider not configured")
                packet = provider(question, image)
                if not isinstance(packet, EvidencePacket):
                    raise TypeError(
                        "providers must return EvidencePacket, not an uncalibrated string"
                    )
            except Exception as exc:
                packet = EvidencePacket(
                    channel=channel,
                    action=action,
                    schema_valid=False,
                    cost=float(self.costs.get(provider_key, 0)),
                    metadata={"failure_type": type(exc).__name__},
                )
            packets[key] = packet
        self.last_packets = packets
        return packets
