# grokking
Reimplementing parts of OpenAI's grokking paper

## Experiments
Below you can see the first successful grokking training run on the modular addition task. I have applied some smoothing to the graph and am not yet sure what causes the spikes, I assume they are because of a numerical instability.

![modular_addtion-2](https://user-images.githubusercontent.com/73826284/192591205-411869b9-c988-4ebe-a2e5-d5f712229954.png)

## References

* [Grokking paper by OpenAI](https://arxiv.org/abs/2201.02177)
* [OpenAI's implementation](https://github.com/openai/grok)
* [PyTorch implementation by Daniel May](https://github.com/danielmamay/grokking)
