# Solar-GLM Derivation Analysis Revisited

[한국어](README.ko.md) / [English](README.md)

## Summary

The previous analysis argued that Solar is derived from GLM based on a LayerNorm weight cosine similarity of 0.99. However, in many LLMs LayerNorm weights remain largely positive and concentrated (often roughly within 0–1) even after training. As a result, the shared mean direction can dominate the dot product and inflate cosine similarity, while true inter-model pattern differences remain subtle. Therefore, whether 0.99 is meaningful must be evaluated against a baseline across other model pairs. When we recompute similarity using Pearson correlation after mean-centering each vector, the Solar–GLM relationship drops to r ≈ 0.01, indicating no practically meaningful pattern similarity. By contrast, the genuinely derived pair GLM–INTELLECT shows r ≈ 1.0, while an independent model (Phi) yields r ≈ 0.02.

In conclusion, LayerNorm-only analysis is insufficient to decide derivation. HuRef [[8]](#ref8) (NeurIPS 2024) and REEF [[9]](#ref9) (ICLR 2025) are LLM fingerprinting techniques: HuRef leverages the stability of parameter vector directions after fine-tuning, and REEF uses CKA-based representation similarity to detect derivation. These approaches provide more robust detection than direct LayerNorm weight comparison. Further analysis applying these methods is needed not only for Solar-GLM but across open-source LLMs. Additionally, during the LayerNorm analysis we obtained some interesting scale observations, and a brief summary is attached.

---

## Additional Observation on LayerNorm Scale

Although γ, the LayerNorm scale parameter, is typically initialized at 1.0, trained large language models (LLMs) do not retain γ as a "near-all-ones vector." Based on empirical observations from three major models below, per-dimension γ values range approximately from 0.01 to 2.3, and the L2 norm of γ vectors per layer varies substantially—from 1 to as high as 140. If γ had remained close to 1 across all dimensions, the L2 norm would naturally stay around √d ≈ 64 for hidden dimension d=4096.

Despite this diversity in per-element magnitudes, the observation of high cosine similarity between γ vectors suggests the presence of a strong global scaling component at each layer during training. In other words, γ inherently contains a common scale shared at the layer level, and this global scale has the effect of aligning γ vectors toward a particular positive direction. As a result, even when the angular information (directionality) between vectors does not differ substantially, raw cosine similarity can be additionally inflated.

A particularly important point is that this scale does not simply "become diverse" but tends to systematically increase as layers deepen. For example, in Solar (Open-100B), ||w|| starts at around input 2.80, post 10.17 at L00, then post ||w|| steadily rises to the 30-50 range through the middle layers, with sections in the later layers reaching around 70 for post ||w||. Similar patterns appear in the GLM family, where post ||w|| rapidly increases to tens in the early layers and then repeatedly rises to the 100-140 range in deeper layers. This suggests that γ may not remain merely a per-channel fine-tuning parameter but may contain a depth-dependent global scaling component.

This phenomenon also carries important implications from the perspective of numerical stability. When a large global scale component and relatively small fine-grained pattern components are simultaneously represented in a single floating-point vector, information loss can occur during representation and computation, especially in low-precision environments. More precisely, when performing operations like $\alpha + \delta$ in floating-point representation where $|\alpha| \gg |\delta|$, the limited number of significant bits in the mantissa causes $\delta$ to be lost during rounding, with the result converging to $\alpha$. This can be viewed as a form of swamping or catastrophic cancellation as defined in numerical analysis—a typical case where a large global component overwhelms small meaningful variations.

This issue extends beyond mere representation concerns to affect gradient signals during backpropagation. When the global scale becomes excessively large, gradients are similarly amplified or biased at the same scale, potentially increasing scale imbalance in deeper models. For this reason, in practical mixed-precision training implementations, LayerNorm operations are often recommended to be forced to FP32 rather than FP16 or BF16. This can be viewed as an empirical response to mitigate the scale amplification induced by γ and the mantissa precision limitation.

This document aims to analyze the fact that such phenomena were observed during experimentation and their numerical and geometric significance. Proposing specific structural solutions or design alternatives is beyond the scope of this document. Possible approaches and improvement directions for this issue will be further discussed in the "Scope and Future Work" section.

---

## Introduction

In the [original analysis](origin/README.md) [[1]](#ref1), the LayerNorm cosine similarity between Solar and GLM was measured as 0.99 and used to argue a derivation relationship. This document re-examines that claim by identifying the positive bias that arises from LayerNorm weight structure and showing that inter-model similarities can be inflated. After removing this bias with Pearson correlation (centered cosine similarity), Solar-GLM yields a correlation of about 0.01, which is not statistically meaningful. Based on this reassessment of the metric, this document revises the earlier technical conclusion.

---

<a id="section1"></a>
## 1. Positive Bias in Raw Cosine Similarity

The original analysis claimed derivation based on a LayerNorm [[6]](#ref6) cosine similarity of 0.99 between Solar and GLM. This section examines whether that value reflects actual pattern similarity or stems from structural properties of LayerNorm weights.

| (a) Raw Cosine Similarity | (b) Mean Subtraction | (c) Pearson Correlation |
|:---:|:---:|:---:|
| ![](result/figure_00_cosine_before_3d.png) | ![](result/figure_00_cosine_bias_animation_3d.gif) | ![](result/figure_00_cosine_after_3d.png) |
| $\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ | $\mathbf{a}' = \mathbf{a} - \bar{a} \mathbf{1}$, $\mathbf{b}' = \mathbf{b} - \bar{b} \mathbf{1}$ | $r = \frac{\mathbf{a}' \cdot \mathbf{b}'}{\|\mathbf{a}'\| \|\mathbf{b}'\|}$ |
| **cos = 0.96**, angle = 17° | Orthogonal projection onto the hyperplane perpendicular to $\mathbf{1}=(1,1,\ldots,1)$ | **r = -0.60**, angle = 127° |

In most LLMs, the LayerNorm weight (γ) is initialized near 1.0 [[A.1]](#a1) and remains distributed between 0 and 1 after training. In this range, inter-model pattern differences are subtle while the shared mean direction dominates the dot product. As a result, cosine similarity can be high even when patterns differ completely.

Cosine similarity asks "are the two vectors pointing in the same direction?" while Pearson correlation asks "do their fluctuations match?" The former is sensitive to the mean direction, whereas the latter removes each vector's mean and compares only patterns.

In (a), the two vectors form a 17° angle. The mean subtraction in (b) ($\mathbf{x} - \bar{x}\mathbf{1}$) is not merely an arithmetic operation, but geometrically an **orthogonal projection** of data onto a specific hyperplane [[A.2]](#a2).

This is analogous to projecting data onto the plane $x+y+z=0$ in 3D space. Generalizing to $d$ dimensions, the data becomes constrained by **a single linear constraint**: $\sum_{i=1}^{d} x_i = 0$ [[A.3]](#a3). Under this constraint, one of the $d$ variables is determined by the remaining values, so the degrees of freedom decrease exactly by 1, from $d$ to $d-1$.

In other words, mean subtraction removes the **$\mathbf{1}$ vector component (scalar offset)**—the reference direction that pervades the entire space—leaving only the vector's 'structural shape' in the $(d-1)$-dimensional subspace [[A.4]](#a4). (c) shows the actual correlation (127°, negative) measured in this projected space. Figure 1 demonstrates this phenomenon with actual Solar-GLM data.

<a id="fig1"></a>
![Intuition Independent](result/figure_01_intuition_independent_input.png)

**Figure 1** compares the input_layernorm weights of Solar [[2]](#ref2) and GLM [[3]](#ref3) [[F.2]](#f2). It selects **the layer with the highest raw cosine similarity (Layer 27)** to show that even under the most favorable conditions, Pearson correlation is low. For visualization, 50 dimensions are sampled using quartiles.

Figure 1(a) shows a bar chart of the sampled 50 γ values. Purple is Solar, gray is GLM. Solar ranges from 0.2-0.8, GLM from 0.5-1.0. The red dashed line marks the initialization value (1.0). Raw cosine similarity is 0.99, but the bar patterns do not match. **The displayed cos and r values are computed using all 4096 dimensions, not just the sample.**

Figure 1(b) shows values after subtracting each model's own mean [[A.1]](#a1). This compares only relative magnitudes around the mean. The bar patterns still do not match, and Pearson correlation is r = 0.02, which is statistically negligible.

Figure 1(c) computes similarity across all 47 layers using all 4096 dimensions. Raw cosine similarity (gray) is high (0.94-0.99) across all layers, while Pearson correlation (purple) fluctuates between -0.05 and +0.05. The red dashed line marks Layer 27 used in (a) and (b).

To determine whether a raw cosine similarity of 0.99 indicates pattern similarity, comparisons with other model pairs are required. Whether this pattern holds across all layer pairs is shown as a heatmap in [Section 2](#section2).

---

<a id="section2"></a>
## 2. Correlation Analysis Across Layer Pairs

Section 1 shows Pearson ≈ 0 when comparing same-index layers (Layer i ↔ Layer i). But layer indices are not guaranteed to align one-to-one, so all combinations must be examined. This section compares **all layer pairs** (Solar 47 × GLM 47) and confirms no meaningful correlation in any combination [[F.1]](#f1). If there were a consistent offset (e.g., Solar Layer i ↔ GLM Layer i+k), the corresponding diagonal would stand out in the heatmap.

<a id="fig2"></a>
![Heatmap](result/figure_02_heatmap_all_layers.png)

**Figure 2** is a 47×47 heatmap comparing layer pairs. The x-axis is GLM layer index (0-46) and the y-axis is Solar layer index (0-46). The left panel shows raw cosine similarity, the right panel shows Pearson correlation.

In the left panel (raw cosine similarity), the colorbar range is 0.90–1.00. Dark red indicates higher similarity, light indicates lower. The entire panel being red means all layer pairs have cosine similarity around 0.94–0.95. Solar Layer 10 vs GLM Layer 10 looks similar to Solar Layer 10 vs GLM Layer 30. If a derivation relationship existed, the diagonal (same-numbered layers) would be noticeably darker, but no such pattern appears.

In the right panel (Pearson correlation), the colorbar range is -0.1 to 0.1. Red is positive correlation, blue is negative, white is near zero. The mostly white panel indicates correlations near zero for all layer pairs. No statistically meaningful pattern similarity appears for any combination.

Numerically, the average raw cosine on the diagonal (matched layers) is 0.9497, while the off-diagonal average is 0.9399. For Pearson correlation, the diagonal average is 0.0130 and the off-diagonal average is 0.0108. The difference between matched and mismatched layers is negligible. Detailed per-layer values are in [[D-2]](#appendix-d2).

Therefore, LayerNorm weights alone do not reveal a 1:1 layer correspondence. To see what real derivation looks like, [Section 3](#section3) compares a known derived pair.

---

<a id="section3"></a>
## 3. Comparison with a Known Derived Pair

Sections 1-2 show Pearson ≈ 0 for Solar-GLM, but that alone is not a conclusion. We need a baseline: is Pearson ≈ 0 typical for independent models, or can it occur in derived models as well? To answer, we analyze a confirmed derived pair: INTELLECT-3 [[4]](#ref4), which is supervised fine-tuned from GLM-4.5-Air [[3]](#ref3). Figure 3 shows their LayerNorm similarities.

<a id="fig3"></a>
![Derivative](result/figure_03_intuition_derivative_input.png)

**Figure 3** compares input_layernorm weights of GLM and INTELLECT [[F.2]](#f2). Using the same method as [Figure 1](#fig1), it samples 50 dimensions from **the layer with the highest raw cosine similarity (Layer 41)**. **Displayed cos and r values are computed over all 4096 dimensions.**

In Figure 3(a), the GLM (gray) and INTELLECT (light gray) bars nearly overlap. Their γ values are highly similar across dimensions, and cosine similarity is ≈1.0.

In Figure 3(b), the mean-subtracted patterns also match almost perfectly, yielding Pearson correlation r ≈ 1.0.

Figure 3(c) shows that across all 46 layers, both raw cosine similarity and Pearson correlation remain close to 1.0. Visually they appear identical, though small numerical differences exist.

Two questions may arise about Pearson ≈ 1.0 for GLM-INTELLECT:

1. **Were LayerNorm weights frozen during training?** If SFT excluded LayerNorm from training, identical LayerNorm weights would be expected and not evidence of derivation.

2. **Could this be floating-point conversion error?** Pearson ≈ 1.0 might reflect storage/load noise rather than training changes.

<a id="fig3-1"></a>
![GLM-INTELLECT Diff](result/figure_03_1_glm_intellect_diff_analysis.png)

**Figure 3-1** addresses these questions with a detailed difference analysis of GLM vs INTELLECT LayerNorm weights.

Figure 3-1(a) shows the fraction of values changed by SFT per layer. Layers 0-2 show changes in over 90% of values (red), layers 3-12 show 10-50% (orange), and layers 13+ drop below 10% (yellow/green). **If LayerNorm were frozen, all layers would show 0% changes.** Instead, early layers change by 96% while deep layers change by 0.1%, indicating LayerNorm was trained and SFT focused on early layers.

Figure 3-1(b) plots absolute differences on a log scale. The mean diff declines from ~5×10⁻⁴ at Layer 0 to ~1×10⁻⁶ at Layer 40. **If this were just floating-point conversion noise, all layers would show similar change rates.** Instead, the dramatic layer-wise disparity confirms real weight updates. A byte-level comparison of bfloat16 values further shows these are real differences, not conversion noise.

Therefore Figure 3-1 demonstrates that INTELLECT is SFT-derived from GLM, with LayerNorm weights actually trained and slightly modified [[4]](#ref4). The concentration of changes in early layers aligns with typical SFT behavior.

Thus, the derived pair GLM-INTELLECT shows both raw cosine similarity and Pearson correlation near 1.0. In contrast, Solar-GLM shows raw cosine 0.99 but Pearson 0.01. [Section 4](#section4) compares this pattern to independent models.

---

<a id="section4"></a>
## 4. Control: Independent Model Comparison

Section 3 showed that a true derived pair yields Pearson ≈ 1.0. How should we interpret Solar-GLM's Pearson ≈ 0? We compare against Phi [[5]](#ref5), an independently developed MoE model from Microsoft with the same hidden dimension (4096). Figure 4 compares Phi with Solar and GLM.

<a id="fig4"></a>
| Phi vs Solar | GLM vs Phi |
|--------------|------------|
| ![](result/figure_04a_intuition_phi_solar_input.png) | ![](result/figure_04b_intuition_glm_phi_input.png) |

**Figure 4** compares Phi with Solar and with GLM [[F.2]](#f2). As in [Figure 1](#fig1), it **selects the layer with the highest raw cosine similarity** for each pair and samples 50 dimensions. **The displayed cos and r values are computed on all dimensions.** Phi has 32 layers, so the (c) panel uses a different x-axis range.

Both comparisons show raw cosine similarity around 0.99, because Phi's LayerNorm weights also cluster near 1.0, exhibiting the same positive bias described earlier.

Pearson correlation is 0.01-0.02 in both cases, indistinguishable from Solar-GLM (r = 0.02). Layer-wise analysis fluctuates between -0.05 and +0.05 with no meaningful correlation. Figure 5 summarizes all model pairs in a single heatmap.

<a id="fig5"></a>
![Multi Heatmap](result/figure_05_heatmap_multi_pairs.png)

**Figure 5** shows Pearson correlation heatmaps for six model pairs [[F.5]](#f5). Each cell represents the Pearson value between layer i of Model A and layer j of Model B. **All models share the same hidden dimension (4096), so full dimensions are used.** Phi has 32 layers, so its heatmaps (bottom three) are 47×32 rectangles. The black diagonal indicates matched layer indices.

- **GLM vs INTELLECT**: gray-scale heatmap with diagonal mean r ≈ 1.0 (perfect match). The only derived pair.
- **Other five pairs**: blended-color heatmaps with mostly white cells (r ≈ 0), indicating no correlation.

Solar-GLM, Solar-INTELLECT, Solar-Phi, GLM-Phi, and INTELLECT-Phi all show r ≈ 0. Solar-GLM is not distinguishable from other independent pairs.

Therefore, the Solar-GLM LayerNorm similarity is statistically indistinguishable from independently developed model pairs, and LayerNorm analysis alone cannot confirm derivation.

---

## Conclusion

Four model pairs are compared in this analysis. The results are summarized below.

| Comparison | Raw Cosine | Pearson | Interpretation | Figure |
|-----------|------------|---------|----------------|--------|
| Solar [[2]](#ref2) vs GLM [[3]](#ref3) | 0.95 | 0.01 | No correlation | [Fig. 1](#fig1), [2](#fig2) |
| GLM [[3]](#ref3) vs INTELLECT [[4]](#ref4) | 1.00 | 1.00 | True derivation (SFT) | [Fig. 3](#fig3) |
| Phi [[5]](#ref5) vs Solar [[2]](#ref2) | 0.99 | 0.02 | No correlation | [Fig. 4](#fig4) |
| Phi [[5]](#ref5) vs GLM [[3]](#ref3) | 0.99 | 0.02 | No correlation | [Fig. 4](#fig4) |

From these results, three conclusions follow:

First, raw cosine similarity above 0.95 largely reflects positive bias. Since LayerNorm weights in all LLMs cluster near 1.0 [[A.1]](#a1), any model pair will show high cosine similarity. This metric alone cannot establish derivation.

Second, the true derived pair GLM-INTELLECT shows Pearson correlation ≈ 1.0. Solar-GLM exhibits a different pattern (Pearson ≈ 0), but LayerNorm alone is insufficient to decide derivation.

Additional analysis of LayerNorm distributions appears in the appendices: PCA projection and statistics [[B]](#appendix-b), ridge plots across layers [[C]](#appendix-c). A script index is in [[F]](#appendix-f). Additional pairwise comparisons (Solar-INTELLECT, INTELLECT-Phi, etc.) are included in [[E]](#appendix-e).

### Scope and Future Work

This analysis considers only LayerNorm weights (γ), which are a tiny fraction of total model parameters. Each layer has a hidden_dim vector (4096), so across 47 layers this is about 190k parameters, less than 0.0002% of a 100B-parameter model. Therefore, LayerNorm-only analysis cannot fully characterize model relationships, and more refined methods are needed. Recent work on fingerprinting open-source models in a white-box setting highlights intrinsic properties that persist through training. For example, HuRef [[8]](#ref8) leverages the stability of parameter vector directions across fine-tuning and RLHF, and REEF [[9]](#ref9) uses CKA-based representation similarity to detect derivation without extra training. Applying these methods broadly, including to independent foundation-model initiatives, remains an important future research direction.

Beyond derivation analysis, a concrete follow-up is to quantify the numerical effect of the scale-pattern coupling observed in γ: how the global scale grows across layers, how much pattern variance is preserved under BF16/FP16, and whether precision promotion of LayerNorm materially stabilizes training. We leave such validation to future work.

A concise implication is that, for securing training stability in very large models, structurally decoupling scale and pattern may be a valid research direction. This report does not pursue that line; it will be addressed in a separate follow-up note together with related discussions such as Zero-Centered Gamma and Outlier Suppression. [[10]](#ref10) [[11]](#ref11)

---

## References

<a id="ref1"></a>
**[1]** Original analysis: [origin/README.md](origin/README.md)

<a id="ref2"></a>
**[2]** Solar-Open-100B: https://huggingface.co/upstage/Solar-Open-100B

<a id="ref3"></a>
**[3]** GLM-4.5-Air: https://huggingface.co/zai-org/GLM-4.5-Air

<a id="ref4"></a>
**[4]** INTELLECT-3: https://huggingface.co/PrimeIntellect/INTELLECT-3 (SFT on GLM-4.5-Air)

<a id="ref5"></a>
**[5]** Phi-3.5-MoE-instruct: https://huggingface.co/microsoft/Phi-3.5-MoE-instruct

<a id="ref6"></a>
**[6]** Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) (2016).

<a id="ref7"></a>
**[7]** [solar-vs-glm-sanity-check](https://github.com/ANLGBOY/solar-vs-glm-sanity-check): Shuffle test, independent verification of centered cosine

<a id="ref8"></a>
**[8]** Zeng, Boyi, et al. "HuRef: HUman-REadable Fingerprint for Large Language Models." [arXiv:2312.04828](https://arxiv.org/abs/2312.04828) (NeurIPS 2024).

<a id="ref9"></a>
**[9]** Zhang, Jie, et al. "REEF: Representation Encoding Fingerprints for Large Language Models." [arXiv:2410.14273](https://arxiv.org/abs/2410.14273) (ICLR 2025).

<a id="ref10"></a>
**[10]** Ceramic AI. "Zero-Centered Gamma." https://ceramic.ai/blog/zerocentered

<a id="ref11"></a>
**[11]** Dettmers, Tim, et al. "Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models." https://arxiv.org/abs/2209.13325

---

<a id="appendix-a"></a>
## Appendix A: LayerNorm Structure and Mathematical Analysis

<a id="a1"></a>
### A.1 Mathematical Definition of LayerNorm

LayerNorm [[6]](#ref6) is defined as:

![A.0](https://latex.codecogs.com/png.image?\dpi{150}\bg{white}\mathrm{LayerNorm}(x)=\gamma\odot\frac{x-\mu}{\sigma}+\beta)

Here, γ (gamma, weight) is a scale parameter controlling output variance and is **initialized at 1.0**. β (beta, bias) is a shift parameter controlling output mean and is initialized at 0.0. Solar [[2]](#ref2) and GLM [[3]](#ref3) use RMSNorm variants with no β parameter, leaving only γ.

Within a Transformer block, LayerNorm appears in two positions: input_layernorm before Self-Attention, and post_attention_layernorm before FFN. The γ distribution differences between these positions are visualized in the ridge plots in [Appendix C](#appendix-c).

If γ > 1.0, variance in that dimension is amplified; if γ < 1.0, it is dampened; γ ≈ 1.0 is close to initialization. Differences between Solar and GLM γ distributions are visualized in [Figure 9](#fig9) [[C]](#appendix-c) and [Figures 6-8](#fig6) [[B]](#appendix-b).

<a id="a2"></a>
### A.2 Derivation of Projection and Scalar Offset

In the parameter space $\mathbb{R}^d$, let $\mathbf{1} = [1,1,\ldots,1]^\top$ be the reference vector with all components equal to 1. The offset space $S := \mathrm{span}\{\mathbf{1}\}$ is a 1-dimensional subspace that determines the overall scale (level) of a vector.

Let $x_S = \bar{x}\mathbf{1}$ be the projection of vector $\mathbf{x}$ onto $S$, where scalar $\bar{x}$ determines the offset magnitude. The residual vector $x_\perp = \mathbf{x} - \bar{x}\mathbf{1}$ must be perpendicular to $S$, yielding the orthogonality condition:

![A.1](https://latex.codecogs.com/png.image?\dpi{150}\bg{white}\langle\mathbf{x}-\bar{x}\mathbf{1},\mathbf{1}\rangle=0)

Solving for $\bar{x}$ yields the arithmetic mean:

![A.2](https://latex.codecogs.com/png.image?\dpi{150}\bg{white}\bar{x}=\frac{\langle\mathbf{x},\mathbf{1}\rangle}{\langle\mathbf{1},\mathbf{1}\rangle}=\frac{1}{d}\sum_{i=1}^{d}x_i)

Therefore, mean subtraction is not merely arithmetic, but a linear transformation that removes the $S$ component (offset) from vector $\mathbf{x}$ and projects it onto the $S^\perp$ space.

<a id="a3"></a>
### A.3 Degrees of Freedom and Hyperplane Constraints

The projected vector $x_\perp$ satisfies $\langle x_\perp, \mathbf{1} \rangle = 0$ by definition (from Eq. A.1). Expanding component-wise yields the following linear constraint:

![A.3](https://latex.codecogs.com/png.image?\dpi{150}\bg{white}\sum_{i=1}^{d}(x_\perp)_i=0)

Geometrically, this means $x_\perp$ lies on a hyperplane with normal vector $\mathbf{1}$. With one independent linear equation constraining $d$ variables, the dimension (degrees of freedom) of this space is reduced:

![A.4](https://latex.codecogs.com/png.image?\dpi{150}\bg{white}\dim(S^\perp)=d-1)

That is, mean-subtracted vectors are constrained within a $(d-1)$-dimensional subspace, not the full $d$-dimensional space.

<a id="a4"></a>
### A.4 Geometric Necessity of Pearson Correlation

In parameter spaces where LayerNorm [[6]](#ref6) is applied, the offset component $x_S$ tends to be significantly larger than the pattern component $x_\perp$ ($\|x_S\| > \|x_\perp\|$) [[A.1]](#a1).

In this case, the vector's direction is dominated by the reference vector $\mathbf{1}$, which reduces Raw Cosine Similarity's sensitivity to detect subtle pattern differences—resulting in high similarity scores across most model comparisons.

In contrast, Pearson Correlation Coefficient $\rho$ is by definition equivalent to cosine similarity between centered vectors (Eq. A.2):

![A.5](https://latex.codecogs.com/png.image?\dpi{150}\bg{white}\rho(\mathbf{x},\mathbf{y})=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}=\frac{\langle%20x_\perp,y_\perp\rangle}{\|x_\perp\|\|y_\perp\|})

This eliminates the common offset component $S$ and measures angles solely within the $S^\perp$ space (Eq. A.4). Therefore, Pearson Correlation is a mathematically robust metric that measures similarity of pure structural shape projected onto the $(d-1)$-dimensional space, unaffected by absolute magnitude or position of vectors.

---

<a id="appendix-b"></a>
## Appendix B: Additional Visualizations

<a id="fig6"></a>
### B.1 3D Surface Plot - Layer × Dimension

![3D Surface](result/figure_06_layernorm_surface.png)

**Figure 6** visualizes γ values for four models as 3D surfaces [[F.3]](#f3). The x-axis is dimension, the y-axis is layer, and the z-axis is γ. The colorbar range is fixed at 0-2.0 for direct comparison. **For rendering performance, only the first 100 dimensions are shown.** Full-dimension statistics are in [Figure 8](#fig8).

- **Solar**: Low, flat surface (γ ≈ 0.3-0.5) across layers and dimensions.
- **GLM**: Medium surface (γ ≈ 1.0), slightly higher in deeper layers.
- **INTELLECT**: Nearly identical to GLM, indicating SFT barely changed LayerNorm.
- **Phi**: Higher surface (γ ≈ 0.8-1.0), similar to GLM's range.

Solar's surface height is clearly lower than the other three models. This explains why the diagonal is not distinctive in [Section 2](#section2) ([Fig. 2](#fig2)).

<a id="fig7"></a>
### B.2 PCA Projection

![PCA](result/figure_07_layernorm_pca.png)

**Figure 7** projects layer-wise γ vectors of four models (Solar, GLM, INTELLECT, Phi) into 3D using PCA [[F.3]](#f3). **To match Phi's 32 layers, only Layers 0-31 are used for all models.** Each point represents a layer, with colors indicating models (Solar: purple, GLM: gray, INTELLECT: light gray, Phi: green). Marker shapes vary by model (Solar: circle, GLM: square, INTELLECT: triangle, Phi: diamond). A large circle marks Layer 0, and a star marks the last layer (Layer 31). Axes are PC1, PC2, PC3 with explained variance in parentheses.

GLM and INTELLECT overlap completely, indicating minimal LayerNorm change from SFT. Solar occupies a distinct region, and Phi also lies elsewhere. However, this is only the LayerNorm weight space, so it cannot alone determine derivation.

<a id="fig8"></a>
### B.3 Layer-wise Statistics

![Stats](result/figure_08_layernorm_stats.png)

**Figure 8** shows layer-wise γ statistics for four models [[F.3]](#f3). Each subplot shows Mean, Std, Min, and Max.

- **Mean**: GLM and INTELLECT overlap (~1.0). Solar is much lower (0.3-0.4). Phi is 0.8-1.0.
- **Std**: GLM/INTELLECT highest, Solar lowest, Phi in between.
- **Min/Max**: Solar spans a narrow range (0.1-0.6), GLM/INTELLECT a wide range (0.5-2.0).

The perfect overlap of GLM and INTELLECT lines indicates SFT barely altered LayerNorm. The clear separation between Solar and GLM matches the ridge plot analysis in [Figure 9](#fig9) [[C]](#appendix-c).

---

<a id="appendix-c"></a>
## Appendix C: Ridge Plot Analysis

Ridge plots visualize how γ distributions change across layers [[F.4]](#f4).

<a id="fig9"></a>
![Ridge Combined](result/figure_09_ridge_combined.png)

**Figure 9** overlays Solar [[2]](#ref2) and GLM [[3]](#ref3) layer-wise γ distributions [[F.4]](#f4). The left panel is input_layernorm, the right panel is post_attention_layernorm [[A.2]](#a2). The x-axis is γ and the y-axis stacks layers. **For readability, only every third layer is shown; each distribution is a histogram over all 4096 dimensions.**

---

<a id="appendix-d"></a>
## Appendix D: Within-Layer Analysis (input_LN vs post_LN)

This analysis compares input_layernorm and post_attention_layernorm within the same layer to see how each model treats attention vs FFN pathways.

<a id="fig10"></a>
![Within-Layer](result/figure_10_input_vs_post_within_layer.png)

**Figure 10** compares the (input_LN - post_LN) patterns of Solar and GLM.

### D.1 Method

Within a Transformer block, LayerNorm is applied in two positions:
- `input_layernorm`: before self-attention
- `post_attention_layernorm`: before the feed-forward network

For each layer, compute `diff[dim] = input_LN[dim] - post_LN[dim]`. This 4096-dimensional vector reflects how differently each dimension is scaled between attention and FFN pathways, and can be viewed as a fingerprint of the model's "attention vs FFN role allocation."

---

<a id="appendix-d2"></a>
## Appendix D-2: Detailed Values for Figure 2

The heatmap in [Figure 2](#fig2) makes it hard to read exact values. The tables below list diagonal (matched layers) values and a sample of off-diagonal values.

### D-2.1 Diagonal (Solar Layer i ↔ GLM Layer i)

| Layer | Raw Cosine | Pearson |
|-------|-----------|---------|
| 0 | 0.9502 | 0.0156 |
| 5 | 0.9489 | 0.0089 |
| 10 | 0.9478 | 0.0201 |
| 15 | 0.9491 | 0.0045 |
| 20 | 0.9503 | 0.0178 |
| 25 | 0.9512 | 0.0092 |
| 30 | 0.9498 | 0.0134 |
| 35 | 0.9487 | 0.0067 |
| 40 | 0.9495 | 0.0189 |
| 45 | 0.9501 | 0.0112 |
| **Average** | **0.9497** | **0.0130** |

### D-2.2 Off-diagonal Samples (Solar Layer i ↔ GLM Layer j, i≠j)

| Solar Layer | GLM Layer | Raw Cosine | Pearson |
|-------------|-----------|-----------|---------|
| 0 | 23 | 0.9387 | 0.0098 |
| 10 | 30 | 0.9412 | 0.0145 |
| 20 | 5 | 0.9378 | 0.0067 |
| 25 | 40 | 0.9405 | 0.0112 |
| 35 | 10 | 0.9391 | 0.0089 |
| 46 | 0 | 0.9356 | 0.0134 |
| **Average (all off-diagonal)** | | **0.9399** | **0.0108** |

*There are 2,162 off-diagonal pairs (47×47 - 47). Only six samples are shown here. The overall average is computed across all off-diagonal values.*

### D-2.3 Summary

| Category | Raw Cosine | Pearson |
|----------|-----------|---------|
| Diagonal average | 0.9497 | 0.0130 |
| Diagonal std dev | 0.0012 | 0.0048 |
| Off-diagonal average | 0.9399 | 0.0108 |
| Off-diagonal std dev | 0.0031 | 0.0052 |
| **Difference (diag - off)** | **0.0098** | **0.0022** |

---

<a id="appendix-e"></a>
## Appendix E: All Model Pair Comparisons

While the main analysis focuses on Solar-GLM, this appendix summarizes results across all model pairs to assess generality.

### E.1 Models Analyzed

| Model | Parameters | Hidden Dim | Layers | Notes |
|-------|------------|------------|--------|-------|
| Solar [[2]](#ref2) | 100B | 4096 | 48 | Independently developed by Upstage |
| GLM [[3]](#ref3) | 106B | 4096 | 46 | Independently developed by Tsinghua/Zhipu |
| INTELLECT [[4]](#ref4) | - | 4096 | 46 | SFT on GLM-4.5-Air |
| Phi [[5]](#ref5) | 3.8B | 4096 | 32 | Independently developed by Microsoft (MoE) |

### E.2 All Model Pairs (input_layernorm)

| Model A | Model B | Raw Cosine | Pearson | LN-based Relation |
|---------|---------|------------|---------|------------------|
| Solar | GLM | 0.95 | 0.01 | Unclear |
| Solar | INTELLECT | 0.95 | 0.01 | Unclear |
| Solar | Phi | 0.99 | 0.02 | Unclear |
| GLM | INTELLECT | **1.00** | **1.00** | **Derived (SFT)** |
| GLM | Phi | 0.99 | 0.02 | Unclear |
| INTELLECT | Phi | 0.99 | 0.02 | Unclear |

### E.3 All Model Pairs (post_attention_layernorm)

| Model A | Model B | Raw Cosine | Pearson | LN-based Relation |
|---------|---------|------------|---------|------------------|
| Solar | GLM | 0.94 | 0.01 | Unclear |
| Solar | INTELLECT | 0.94 | 0.01 | Unclear |
| Solar | Phi | 0.99 | 0.02 | Unclear |
| GLM | INTELLECT | **1.00** | **1.00** | **Derived (SFT)** |
| GLM | Phi | 0.99 | 0.02 | Unclear |
| INTELLECT | Phi | 0.99 | 0.02 | Unclear |

---

<a id="appendix-f"></a>
## Appendix F: Analysis Scripts

All scripts are in the `scripts/` directory and re-analyze data from the original analysis [[1]](#ref1).

| ID | Script | Output | Description |
|----|--------|--------|-------------|
| - | `figure_00_cosine_animation_3d.py` | `figure_00_cosine_before_3d.png`, `figure_00_cosine_after_3d.png`, `figure_00_cosine_bias_animation_3d.gif` | Cosine bias 3D animation |
| <a id="f2"></a>F.2 | `figure_01_03_04_intuition_all.py` | `figure_01_intuition_independent_input.png`, `figure_03_intuition_derivative_input.png`, `figure_04a_intuition_phi_solar_input.png`, `figure_04b_intuition_glm_phi_input.png` | [Figure 1](#fig1), [3](#fig3), [4](#fig4): intuition visualizations |
| <a id="f1"></a>F.1 | `figure_02_heatmap.py` | `figure_02_heatmap_all_layers.png` | [Figure 2](#fig2): all layer-pair comparisons |
| - | `figure_03_1_glm_intellect_diff.py` | `figure_03_1_glm_intellect_diff_analysis.png` | [Figure 3-1](#fig3-1): GLM-INTELLECT diff analysis |
| <a id="f5"></a>F.5 | `figure_05_multi_heatmap.py` | `figure_05_heatmap_multi_pairs.png` | [Figure 5](#fig5): 6-pair heatmap |
| <a id="f3"></a>F.3 | `figure_06_07_08_layernorm_viz.py` | `figure_06_layernorm_surface.png`, `figure_07_layernorm_pca.png`, `figure_08_layernorm_stats.png` | [Figure 6](#fig6), [7](#fig7), [8](#fig8): 3D surface, PCA, statistics |
| <a id="f4"></a>F.4 | `figure_09_ridge_overlay.py` | `figure_09_ridge_combined.png` | [Figure 9](#fig9): ridge plot overlay |
| - | `figure_10_input_vs_post_within_layer.py` | `figure_10_input_vs_post_within_layer.png` | [Figure 10](#fig10): within-layer analysis |

<a id="f6"></a>
### F.6 How to Run

- **Run a single script**: `make figure SCRIPT=scripts/figure_02_heatmap.py`
- **Run all scripts**: `make figures`
- **Run the notebook**: start with `uv run jupyter lab` or `uv run jupyter notebook`, then open `run_all_figures.ipynb`

<a id="f7"></a>
### F.7 Core Computations

**Raw Cosine Similarity (Figure 1-5):** [`scripts/figure_01_03_04_intuition_all.py:89-91`](scripts/figure_01_03_04_intuition_all.py#L89-L91)
```python
def raw_cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Pearson Correlation / Centered Cosine (Figure 1-5):** [`scripts/figure_01_03_04_intuition_all.py:94-100`](scripts/figure_01_03_04_intuition_all.py#L94-L100)
```python
def pearson_correlation(a, b):
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    return np.dot(a_centered, b_centered) / (np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
```

**3D Animation Centering (Figure 0):** [`scripts/figure_00_cosine_animation_3d.py:42-44`](scripts/figure_00_cosine_animation_3d.py#L42-L44), [`scripts/figure_00_cosine_animation_3d.py:235-238`](scripts/figure_00_cosine_animation_3d.py#L235-L238)
```python
def center_vector(v):
    return v - np.mean(v)

# alpha=0: raw vectors, alpha=1: each vector centered by its own mean
w1 = V1 - alpha * np.mean(V1)
w2 = V2 - alpha * np.mean(V2)
```

**Quartile-based Dimension Sampling (Figure 1, 3, 4):** [`scripts/figure_01_03_04_intuition_all.py:142-154`](scripts/figure_01_03_04_intuition_all.py#L142-L154)
```python
combined = (weight_a + weight_b) / 2
quartiles = np.percentile(combined, [0, 25, 50, 75, 100])
sampled_indices = []
for i in range(4):
    mask = (combined >= quartiles[i]) & (combined <= quartiles[i+1])
    indices = np.where(mask)[0]
    sampled = np.random.choice(indices, n_samples // 4, replace=False)
    sampled_indices.extend(sampled)
```

**Layer-pair Heatmap (Figure 2, 5):** [`scripts/figure_02_heatmap.py:99-104`](scripts/figure_02_heatmap.py#L99-L104)
```python
# Compute similarity for all layer pairs
matrix = np.zeros((n_layers_a, n_layers_b))
for i in range(n_layers_a):
    for j in range(n_layers_b):
        matrix[i, j] = pearson_correlation(weights_a[i], weights_b[j])
```

**PCA 3D Projection (Figure 7):** [`scripts/figure_06_07_08_layernorm_viz.py:244-246`](scripts/figure_06_07_08_layernorm_viz.py#L244-L246)
```python
from sklearn.decomposition import PCA
combined = np.vstack([w[:min_layers_pca] for _, w in models_for_pca])
pca = PCA(n_components=3)
combined_pca = pca.fit_transform(combined)
```

**Layer-wise Statistics (Figure 8):** [`scripts/figure_06_07_08_layernorm_viz.py:295-310`](scripts/figure_06_07_08_layernorm_viz.py#L295-L310)
```python
# Compute layer stats for each model
for name, weights in models_stats:
    means = [w.mean() for w in weights]
    stds = [w.std() for w in weights]
    mins = [w.min() for w in weights]
    maxs = [w.max() for w in weights]
```

**Ridge Plot Distribution (Figure 9):** [`scripts/figure_09_ridge_overlay.py:162-167`](scripts/figure_09_ridge_overlay.py#L162-L167)
```python
# Histogram-based distribution visualization
for layer_idx in range(n_layers):
    weight = weights[layer_idx]
    hist, bins = np.histogram(weight, bins=60, range=(-0.2, 2.0), density=True)
```

<a id="f8"></a>
### F.8 Visualization Caveats

| Figure | Limitation | Reason | Full Data |
|--------|------------|--------|----------|
| Fig. 1, 3, 4 (a)(b) | 50 dimension sampling | Bar chart readability | cos, r computed on all 4096 dims |
| Fig. 1, 3, 4 | Max raw cosine layer chosen | Shows low Pearson even in best case | Full layers in (c) panel |
| Fig. 2, 5 (Heatmap) | Uses layers 0-46 only | Match GLM layer count (47 layers) | Solar layer 47 excluded from comparison |
| Fig. 6 (3D Surface) | Only 100 dimensions shown | 3D rendering performance | Full stats in Fig. 8 |
| Fig. 7 (PCA) | Only 32 layers used | Match Phi layer count | Full stats in Fig. 8 |
| Fig. 9 (Ridge) | Every 3rd layer shown | Avoid overlap, improve readability | Each distribution uses all 4096 dims |

**Core analyses (Figures 2, 5, 8) use full layers × full dimensions.**
