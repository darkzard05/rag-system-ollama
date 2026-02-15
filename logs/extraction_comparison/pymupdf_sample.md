## CM3: A CAUSAL MASKED MULTIMODAL MODEL OF
### THE INTERNET

**Armen Aghajanyan, Bernie Huang** _[∗]_ **, Candace Ross** _[∗]_ **, Vlad Karpukhin** _[∗]_ **, Hu Xu** _[∗]_ **, Naman Goyal,**
**Dmytro Okhonko, Mandar Joshi, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer**
Facebook AI Research
_{_ armenag,berniehuang,ccross,vladk,huxu,naman _}_ @fb.com
_{_ oxo,mandarj,gghosh,mikelewis,lsz _}_ @fb.com


ABSTRACT


We introduce CM3, a family of causally masked generative models trained over a
large corpus of structured multi-modal documents that can contain both text and
image tokens. Our new causally masked approach generates tokens left to right
while also masking out a small number of long token spans that are generated at
the end of the string, instead of their original positions. The casual masking object provides a type of hybrid of the more common causal and masked language
models, by enabling full generative modeling while also providing bidirectional
context when generating the masked spans. We train causally masked languageimage models on large-scale web and Wikipedia articles, where each document
contains all of the text, hypertext markup, hyperlinks, and image tokens (from a
VQVAE-GAN), provided in the order they appear in the original HTML source
(before masking). The resulting CM3 models can generate rich structured, multimodal outputs while conditioning on arbitrary masked document contexts, and
thereby implicitly learn a wide range of text, image, and cross modal tasks. They
can be prompted to recover, in a zero-shot fashion, the functionality of models
such as DALL-E, GENRE, and HTLM (Ramesh et al., 2021; De Cao et al., 2020;
Aghajanyan et al., 2021). We set the new state-of-the-art in zero-shot summarization, entity linking, and entity disambiguation while maintaining competitive
performance in the fine-tuning setting. We can generate images unconditionally,
conditioned on text (like DALL-E) and do captioning all in a zero-shot setting
with a single model.


1 INTRODUCTION


Recent advancements in large-scale generative sequence modeling have significantly improved zeroshot performance on multiple modalities, including text Brown et al. (2020); Fabbri et al. (2020);
Aghajanyan et al. (2021) and images Ramesh et al. (2021). Recent work has also shown how to
use document structure, e.g., as provided by HTML web markup, to enable more effective zero-shot
prompting for text-only tasks (Aghajanyan et al., 2021). In this paper, we show it is possible to learn
multi-modal document-structured generative models, to jointly represent formatted hypertext and
images as they naturally co-occur within full document contexts.


We introduce CM3, a family of causally masked generative models trained over a large corpus of
structured multi-modal documents. Causally masked models generate tokens left to right, just like
a causal language model, but also mask out a small number of long token spans, which are then
generated at the end of the string instead of their original positions. This provides a new hybrid of
causal and masked language models, enabling full generative modeling with bidirectional context.
For example, it can also be used in our setting to infill complete images or larger structured text
sections, conditioned on the rest of the document.


We train CM3 models on close to a terabyte of web-based data following Aghajanyan et al. (2021),
extended to include images through VQVAE-GAN tokens (Esser et al., 2021) and additional hypertext link structure. This data is in strong contrast to previous methods that were either uni-modal or


_∗_ Equal Contribution for Second Author


1




carefully curated the image-text alignment (e.g., for image captioning Radford et al. (2021); Ramesh
et al. (2021)). We train a 2.7 billion and 13 billion causally masked model on this data which we
call CM3-Medium and CM3-Large respectively.


Extensive experiments demonstrate that these models are able to perform a wide range of zero-shot
uni- and cross-modal tasks. We show both qualitatively and quantitatively that CM3 can be prompted
for non-trivial image generation, similar to that of DALL-E. We also show that CM3 models are capable of improving over state-of-the-art zero-shot summarization, entity linking, entity disambiguation, highlighting the structure that comes from the hypertext during training. Finally, we show that
by fine-tuning CM3 we set the new state-of-the-art for entity linking and entity disambiguation in
general.


To summarize, our contributions include:


    - We present the first hyper-text language-image model, trained on close to a Terabyte of
multi-modal simplified HTML data from the common crawl.

    - We present the causally masked objective, a hybrid of causal and masked language models
that allows for bidirectional context control during generative mask infilling.

    - We demonstrate consistently strong transfer from CM3 to a range of uni-modal and multimodal tasks at differing supervision