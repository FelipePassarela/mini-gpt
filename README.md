# mini-gpt

## Transformer block demo

Run the demo from the project root.

```sh
uv run mini-gpt
```

The demo passes a fixed sequence of four three-dimensional token vectors through a
causal `TransformerBlock` and prints a `(4, 3)` tensor of hidden states. The block
uses randomly initialized weights, so the values can vary between runs.
