# Testing the Meta In-Context Learning Capabilities of Transformers

This repository explores the behaviour of _in-context learning_, the same mechanism that GPT models display by adjusting their predictions based on the additional data given _in context_. Some papers compare the behavior of in-context learning to gradient descent (von Oswald et al., 2022a), where each transformer layer corresponds to a gradient descent step which is implicitly performed in the model. This behavior shows up when the transformer is trained in a meta-learning fashion, by optimizing on a distribution of regression datasets.

In `in-context learning mechanism.ipynb` I explore a reimplemented a simple version the transformer in von Oswald et al., 2022a. To explore the similarities to gradient descent and explore the structures of the different projections (von Oswald et al., 2022b).
<p>
    <a href= "https://colab.research.google.com/github/RobvanGastel/meta-in-context-learning/blob/main/in-context learning mechanism.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>

In `general-purpose in-context learner.ipynb` I continue the exploration of in-context learning to see how it can be used more explicitly for meta-learning 
(Kirsch et al., 2022). The interesting element of this method is that by augmenting the training set and meta-learning it generalizes to out-of-distribution datasets in few-shot learning setting. _Finishing up the latest experiments_
<p>
    <a href= "https://colab.research.google.com/github/RobvanGastel/meta-in-context-learning/blob/main/general-purpose in-context learner.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>

In recent work (Minegishi et al. 2025), they extend the analysis of small-scale transformer models for meta in-context learning and hypothesize that the induction head alone can not explain all of the (meta) in-context learning that happens within LLMs. There are also works exploring in-context learning on larger transformers, most prominent example is the Titans architecture (Behrouz, Zhong, and Mirrokni 2024). 

## Setup
Install the packages using the `requirements.txt` file.

```bash
# using conda
conda create --name icl python=3.11
conda activate icl
# Install the package for meta_icl imports
pip install -e .
# Or run a notebook directly
```

## References
Kirsch, L., Harrison, J., Sohl-Dickstein, J., & Metz, L. (2022). General-Purpose In-Context Learning by Meta-Learning Transformers (arXiv:2212.04458). arXiv. http://arxiv.org/abs/2212.04458

Han, S., Song, J., Gore, J., & Agrawal, P. (2024). Emergence of Abstractions: Concept Encoding and Decoding Mechanism for In-Context Learning in Transformers (arXiv:2412.12276). arXiv. https://doi.org/10.48550/arXiv.2412.12276

von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., & Vladymyrov, M. (2022a, December 15). Transformers learn in-context by gradient descent. arXiv.Org. https://arxiv.org/abs/2212.07677v2

Olsson, et al., "In-context Learning and Induction Heads", Transformer Circuits Thread, 2022b. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html.

Minegishi, Gouki, Hiroki Furuta, Shohei Taniguchi, Yusuke Iwasawa, and Yutaka Matsuo. 2025. “In-Context Meta Learning Induces Multi-Phase Circuit Emergence.” https://openreview.net/forum?id=LNMfzv8TNb&referrer=%5Bthe%20profile%20of%20Yutaka%20Matsuo%5D(%2Fprofile%3Fid%3D~Yutaka_Matsuo1) (April 17, 2025).

Behrouz, Ali, Peilin Zhong, and Vahab Mirrokni. 2024. “Titans: Learning to Memorize at Test Time.” doi:10.48550/arXiv.2501.00663.
