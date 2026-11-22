# mini-gpt

## Token embeddings and Transformer block demo

Run the demo from the project root.

```sh
uv run mini-gpt
```

The demo starts with a fixed sequence of four integer token IDs.
`TokenAndPositionEmbeddings` adds a learned token vector and a learned position
vector for each ID. A causal `TransformerBlock` transforms those vectors into
hidden states.

The command prints the input IDs and a `(4, 3)` tensor with one three-dimensional
hidden state per token. The weights are randomly initialized, so the values can
vary between runs.
