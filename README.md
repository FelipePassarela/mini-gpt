# mini-gpt

## Next-token prediction demo

Run the demo from the project root.

```sh
uv run mini-gpt
```

`MiniGPT()` processes the prompt `[1, 0, 6, 5]` and returns logits with shape
`(4, 65)`, one score per vocabulary token at each input position. The demo uses
the logits from the last position, normalizes them across the vocabulary, and
prints the five most likely next token IDs with their probabilities.

The weights are randomly initialized, so the results can vary between runs.
