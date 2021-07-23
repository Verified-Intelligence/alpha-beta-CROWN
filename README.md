α,β-CROWN: Complete and Incomplete Neural Network Verification with Efficient Bound Propagations
----------------------

α,β-CROWN is the winning verifier in [VNN-COMP 2021](https://sites.google.com/view/vnn2021). It combines our existing efforts on neural network verification:

* **CROWN** ([Zhang et al. NeurIPS 2018](https://arxiv.org/pdf/1811.00866.pdf)) is a very efficient bound propagation based verification algorithm. CROWN propagates a linear inequality backwards through the network and utilizes linear bounds to relax activation functions.

* **LiRPA** ([Xu et al., NeurIPS 2020](https://arxiv.org/pdf/2002.12920.pdf)) is a generalization of CROWN on general computational graphs and we also provide an efficient GPU implementation, the [auto\_LiRPA](https://github.com/KaidiXu/auto_LiRPA) library.

* The **"convex relaxation barrier"** ([Salman et al., NeurIPS 2019](https://arxiv.org/pdf/1902.08722)) paper concludes that optimizing the ReLU relaxation allows CROWN (referred to as a "greedy" primal space solver) to achieve the same solution as linear programming (LP) based verifiers.

* **α-CROWN** (sometimes referred to as optimized CROWN or optimized LiRPA) is used in the Fast-and-Complete verifier ([Xu et al., ICLR 2021](https://arxiv.org/pdf/2011.13824.pdf)), which jointly optimizes intermediate layer bounds and final layer bounds in CROWN via variable α. α-CROWN typically has greater power than LP since LP cannot cheaply tighten intermediate layer bounds.

* **β-CROWN** ([Wang et al. 2021](https://arxiv.org/pdf/2103.06624.pdf)) incorporates split constraints in branch and bound (BaB) into the CROWN bound propagation procedure via an additional optimizable parameter β. The combination of efficient and GPU accelerated bound propagation with branch and bound produces a powerful and scalable neural network verifier.

The original code used in the competition can be found in the `vnncomp2021` branch, and we are currently preparing a new release with code cleanups and better documentation.

