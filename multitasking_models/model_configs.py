from dataclasses import dataclass, field

@dataclass(frozen=True)
class ModelConfig(): 
    model_name: str
    rnn_layers: int
    out_dim: int
    rnn_hidden_dim: int = field(default=128, init=True)


config = ModelConfig(model_name='sbert', rnn_layers=1, out_dim=20)

config.rnn_hidden_dim = 1

print(config)