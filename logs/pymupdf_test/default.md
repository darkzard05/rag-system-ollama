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

    - We demonstrate consistently strong transfer from CM3 to a range of uni-modal and multimodal tasks at differing supervision levels, including stating state-of-the-art on entity disambiguation and zero-shot summarization.

    - We release all code and models to support future CM3 research.


2 CAUSALLY MASKED OBJECTIVE


Traditional approaches to pre-training have focused on mixing the architectural choices (i.e.,
encoder-only, decoder-only, encoder-decoder) with objective choices (i.e., masking, causal language
modeling). For example, masked encoder-only models such as BERT (Devlin et al., 2018) and
RoBERTa (Liu et al., 2019) excel in non-generative fine-tuning tasks. Masked encoder-decoder
models such as BART (Lewis et al., 2019) and T5 (Raffel et al., 2019) excel in both discriminative
and generative fine-tuning. Brown et al. (2020) on the other hand, showed that causal language models (de-facto, decoder only) are capable of non-trivial performance without the need of fine-tuning
by simply prompting with appropriate string to control the generated outputs Radford et al. (2019);
Brown et al. (2020); Artetxe et al. (2021).


There are pros and cons to both masked and causal language modeling in the context of prompting.
Masking offers the critical ability to encode bi-directionality within the prompts at the cost of only
decoding roughly 15% of the tokens of the input sequence during training (Devlin et al., 2018;
Liu et al., 2019; Lewis et al., 2019). Conversely, decoder-only causal language models decode every
token in the input sequence in the training but are typically limited to left-only contexts. Empirically,
more work has also been done on scaling causal decoder-only rather than their counterparts.


In an effort to get most of the best of both worlds, we introduce a novel objective that combines
the benefit of per-token generation with optional bi-directionality specifically tailored to prompting.
For a document of size _s_ we select _n ∼_ Clamp(Poisson(1) _,_ 1 _,_ 16) masks and for each of those
masks we select span _m ∼_ ( _Uniform_ (0 _, s_ ) _, Uniform_ (0 _, s_ )) which does not intersect with any
other _m_ . These values are chosen to, on average, select relatively few relatively long spans, which
we expect will allow the model to learn to infill long spans. We then order these masks by the order
that they appear in the source document, replace the span of the mask in the source document with
an enumerated mask token (i.e., <mask:0>, <mask:1>), and move the masked spans to the end
of the document followed by a unique end of document token.


Figure 1 shows the complete process.


We also augment the standard cross-entropy loss to weigh the loss of predicting mask tokens as
0, as they are placed at random locations which carry no information to the underlying sequence
modeling objective.


The complete array of benefits will become more apparent when designing prompts for uni/crossmodal tasks in § 4. However, at the core, the causally masked objective can do causal language
modeling while optionally allowing for bidirectionality when needed.


2


**Causally**

**Masked**
**Language**

**Model**




|Monte Melkonian was a left-wing <mask:0> nationalist militant . <mask:0> <a href= Armenian _nationalism >|Col2|Col3|
|---|---|---|
|Monte|>|>|



Masked

|<a href= Armenian _nationalism >|Col2|Col3|
|---|---|---|
|<a|>|>|

Language Monte Melkonian was a left-wing <mask> nationalist militant .

Model

|Monte Melkonian was a left-wing <a href= Armenian _nationalism > nationalist militant .|Col2|Col3|
|---|---|---|
|Monte|.|.|



Figure 1: A visual representation of various language modeling objectives as well as our proposed
causal language modeling objective with a single mask ( _n_ = 1). Given the left-to-right nature of
causal language models (bottom row) we would not be able to generate the Wikipedia entity link
highlighted in orange.


3 CM3


Aghajanyan et al. (2021) used structured documents for text-only pre-training with strong zero-shot
performance. **C** ausally- **M** asked **M** ultimodal **M** odeling (CM3) extends this work by modeling full
document structure including images and hypertext links. Furthermore, we move away from the
BART-like objective of Aghajanyan et al. (2021) to use our new causally masked objective with
decoder-only models.


3.1 DATA


Following Aghajanyan et al. (2021) we aim to implement a transform over HTML documents to
extract out to minimal-HTML, i.e., the minimal set of text that is semantically relevant for end tasks.


Birhane et al. (2021) gave in-depth criticisms of Common Crawl based multi-modal datasets and
showed the existence of highly problematic examples (i.e., explicit images and text pairs of rape,
pornography, and ethnic slurs). Given these severe ethical concerns, we opt-out of processing all
of Common Crawl and instead opt into using a subset of the Common Crawl News (CC-NEWS)
dataset and all of English Wikipedia.


Given a valid HTML DOM [1] per document, we run several passes to strip down the DOM to the
elements of maximal semantic value. We first remove all elements which do not contain textual elements. We also filter out all _headers_, _footers_, _copyrights_, _forms_, _dialog boxes_ and _iFrames_ . We fold
consecutive <div> elements into a singular <div> element with merged attributes. Furthermore
we strip all the attributes from every element which are not derived from structured graphs such as
OpenGraph, Schema and Twitter.


For every <img> tag in the document with a valid src attribute URL, we download the image,
resize to 256x256 pixels with random cropping and then tokenize it with VQVAE-GAN from Esser
et al. (2021). This amounts to 256 tokens for every image. We then insert the string value of the
tokens joined with a space back into the src attribute.


We do not place any restrictions on the number of images or their locations. We present a set of
high-level statistics in Table 1.


Documents (Million) Size (GB) Unique Images (Million) Tokens (Billion)


CC-NEWS 45 460 18 121
En-Wikipedia 16 383 7 102


Total 61 843 25 223


Table 1: High level statistics of the data used to train CM3.


1The DOM or Document Object Model is an interface that treats an HTML document as a tree structure
wherein each node is an object representing a part of the document.


3


For experimentation, we create two test sets from each data source with 10,000 unique documents
for each. We de-duplicated our test sets to ensure no overlap between test and train sets to the best
of our abilities.


3.2 SIZE HINTS


Aghajanyan et al. (2021) introduced the concept of size hints which allows the user to guide the
model during sample generation through token conditioning. Specifically, HTLM inserts a probabilistic estimate of the size of the mask as a token post the mask token (e.g., <mask>12 for a
probabilistic size of 12). For CM3, we noticed that size-hints degraded not only end-perplexity but
also the zero-shot performance on a significant set of evaluation tests.


We also note that we can implicitly give a size hint during mask generation for a single mask by
asking the model to generate causally max ~~s~~ equence ~~l~~ ength - size ~~h~~ int tokens before
placing the secondary <mask:0> token.


3.3 TRAINING


We train 4 models; 125M, 800M, 2.7B, and 13B parameters. The purpose of the two smaller models
was to establish basic hyper-parameters that are viable for the causally masked language modeling
objective and therefore were under-trained. However, all downstream tasks will be evaluated with
our 2.7B model (CM3-Medium) and our 13B model (CM3-Large). HTLM-Medium was trained on
240 V100 GPU for 28 days, while HTLM-Large was trained on 384 A100 GPU for 24 days. Our
implementation was in PyTorch (Paszke et al., 2019) using fairseq (Ott et al., 2019) and fairscale
(Baines et al., 2021). For every model, our per GPU batch size was 8, with a maximum token
sequence length of 2048. We use the polynomial decay learning rate scheduler available in Paszke
et al. (2019) with 1500 warmup updates. We clipped the gradient norms to 1.0 and used the Adam
optimizer with _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 98 (Kingma & Ba, 2014). We defer our model architecture
description to § A.1.


3.4 SCALING LAWS


Our training setting has a couple of new parameters that can impact the traditional scaling laws
of causal language models. The multi-modal nature of our proposed model breaks the standard
assumptions of token distributionality. Traditionally language tokens are said to follow a Zipfian
distribution (Piantadosi, 2014), while image tokens are strictly uniform (see § A.2). Furthermore, the
unrestricted locations of the images and text introduce unpredictable complexity. Lastly, although
we are still computing the joint probability of the document, we do so in a roundabout way through
shuffling of the document via the causally masked objective. These fundamental differences warrant
a quick look into the scaling laws of CM3.















Figure 2: Basic perplexity based scaling laws for the proposed CM3 objective and training set-up.


We present the various perplexity curves for the four models of varying sizes we trained. Given that
our models were trained on various hardware set-ups, we normalize the training time by linearly
scaling each experiment timing to 256 GPU. Most importantly, we see healthy scaling, similar to


4


Kaplan et al. (2020) without any pathological cases, implying there is still a good amount of gains
to be achieved with further scaling. An in-depth analysis of the scaling laws of the causally masked
objective is outside this current work’s scope and will be considered for future work.


4 ZERO/FEW-SHOT PROMPTING


4.1 IMAGE MODALITY


Although we do not train on pure image documents, CM3 can still operate over image tasks. To do
so, we must cleverly describe the task through a textual prompt, using the <img> tag.


4.1.1 UNCONDITIONAL IMAGE GENERATION


To sample from the distribution of images available to CM3, we can simply ask the model to produce
the next set of tokens after the following prompt: <img.


Interestingly enough, CM3 prefers to first generate a short description of the image through the alt
attribute and then generate the image tokens via the src attribute. We can force the model to directly
generate image tokens without first giving a description with the following prompt: <img src=".
We consider both prompts to test unconditional image generation since we do not condition the
image generation but rather the model self-conditions.


We sample according to the distribution of the model without altering the temperature. We present
a sample of non-cherry picked examples in Figure 3.


<img



(a) A mountain of
olive trees on the
way to Cabo de la
Vela



(b) Spain Europa
Amenacer Winter



(c) blog TIGI Bed
Head Tie Dye
Spray Hair Spray
Hairspray ml



(d) birthday invitation printable
christmas gift for
birthday party
Printable Template



<img src="


Figure 3: Four samples for two of the prompts we proposed for unconditional image generation for
CM3-Large. For the self-captioned images we place the respective caption under the image. Results
were selected at random, with no cherry picking.


The model is more than capable of generating coherent images. We note that via this prompting, we
can recover the full functionality of the DALL-E model proposed in Ramesh et al. (2021). Interestingly enough, we see qualitative improvements with allowing the model to free generate a caption
prior to generating.


We continue by doing an empirical study of the unconditional generation of CM3, by generating 30k
samples without textual conditioning and calculating the Fr´echet Inception Distance (FID, Heusel
et al. (2017)) over MS-COCO, following the methodology proposed in Nichol et al. (2021) (Lin
et al., 2014). We present our results in the unified table showing FID calculations in Table 2. Without any textual conditioning and without explicitly optimizing for either MS-COCO or generation


5


(unlike other benchmarks in the table) CM3 Large approaches the FID performance of modern Generative Adversarial Networks (GAN).


4.1.2 IMAGE IN-FILLING


Unlike DALL-E, which leverages left-to-right language modeling objective to model languageimage tokens, CM3 with the proposed causally masked language modeling makes it possible to
condition contiguous sections of an image on the surrounding context for image in-filling. Specifically, CM3 can infill images with the following prompt:


**Infilling Prompt:** <img src=" _{_ prefix _}_ <mask:0> _{_ postfix _}_ "><mask:0>


Using the same decoding strategies described in § 4.1.1 we generate unconditional infilled images
with only CM3-Large and present qualitative results in Figure 4. Overall we see that CM3-Large is
capable of generating semantically coherent infills even without grounding in text.


4.2 TEXT-IMAGE MODALITY


4.2.1 CONDITIONAL IMAGE IN-FILLING


Additionally, CM3 can further perform image in-filling condition on the additional text context. This
can be achieved by slightly augmenting the prompt as follows:


**Conditional Infilling Prompt:**


<img alt="Photo: _{_ text _}_ " src=" _{_ prefix _}_ <mask:0> _{_ postfix _}_ "><mask:0>


We show qualitative results in Figure 4. Immediately we notice the substantial improvement in the
generated image when grounded in ground truth text vs. unconditional image-infilling.


4.2.2 CONDITIONAL IMAGE GENERATION


We can do conditional text generation using CM3 similar to DALL-E by using a proper prompt.
Specifically by conditioning using the alt attribute of the img tag.


**Conditional Generation Prompt:** <img alt=" _{_ prompt _}_


We present qualitative conditional image generation results in Figure 5. Specifically, we sample
32 images for every prompt given and re-rank using CLIP to get the top-4 images (Radford et al.,
2021). Overall we see that CM3 can generate recognizable images of the input text. There are still
failure cases, such as the second image in the second prompt, where the model easily generates a
landscape but forgets to generate the red car. The third prompt, CM3, is incapable of drawing the
face of a sheep while getting the general body and texture correct.


We note that CM3 trains with an order of magnitude less unique images than DALL-E, and the subset
of images available to CM3 are the images available in news and Wikipedia articles; therefore, CM3
does not generate fictional images well. That being said, casting a larger pool for CLIP selection by
randomly sampling a larger set qualitatively fixes some of these issues.


For quantitative analysis, we compute FID on MS-COCO following the methodology provided by
Nichol et al. (2021). Specifically, we sample 30k samples conditioned on MS-COCO captions. For
all models, we use a temperature of 0.85 and do straightforward sampling.


We present our FID results on MS-COCO 256x256 in Table 2. In general CM3 is capable of generating semantically coherent images on-par with modern GANs. Furthermore, our conditional CM3Large model approaches the performance of the DALL-E model while using an order of magnitude
fewer data.


4.2.3 CAPTIONING


We next look at the dual-task to conditional image generation and image captioning. We can prompt
CM3 to do zero-shot image captioning by asking the model to generate either the alt or title


6


group of people
windsurfing over
the beach and
water in the
ocean.


the wooden park
benches are
painted dark
purple.


some bread is on
a plate with
jam, an apple,
yogurt and
orange juice.


a nice looking
hotel room with
a neatly done
bed, coffee
table, and a
chair.


Source Image Masked/Tokenized Image CM3-Infilling-U CM3-Infilling-C Ground Truth


Figure 4: We provide qualitative samples for zero-shot image-infilling using the CM3-Large model
using the aforementioned prompts. CM3-Infilling-U refers to infilling without conditioning on text
while CM3-Infilling-C refers to conditioning on the ground truth text.


attributes of a properly set <img> tag. Due to attributes always appearing in alphabetical order in
order to generate alt attribute (which appears before src), we need to use the masking capabilities
of CM3.


**Captioning Masked Prompt #1:** <img alt="Photo: A photo
taken of<mask:0>" src=" _{_ image _}_ ">


**Captioning Causal Prompt #1:** <img src=" _{_ image _}_ "
title="Photo: A photo taken of


We have two methods of generating captions given the above prompts. First, the relatively inexpensive method involves running beam-search with a beam size of 5 over the proposed prompts. For a
single image, we run both available prompts and select the sequence, which minimizes the respective CM3 perplexity. The second method is much more costly and requires sampling 128 captions for
every image (we note that this is cheaper than image generation since image generation requires the
minimal generation of 256 image tokens while captioning is usually on the order of a dozen tokens).
We then use CLIP from Radford et al. (2021) to get the top ranking caption. We note that non-trivial
captioning behavior was only exhibited in CM3-Large model; therefore, all evaluations will consider
this singular model.


We provide a qualitative example in Figure 6, sourcing images and ground truth captions from
MS-COCO (Lin et al., 2014). We see that CM3 is capable of generating non-trivial semantically
coherent captions. That being said, most failure cases of our proposed zero-shot captioning are due


7


an armchair
in the shape
of an
avocado. an
armchair
imitating an
avocado.


A red car in
the
mountains.


Photo: A
sheep in
snowy Artsakh


Photo: An
Armenian
church during
springtime
with clear
skies


Figure 5: Four samples for four of the prompts using the conditional image generation prompt with
CM3-Large. Results were selected by CLIP from a candidate set of 32 samples.


Model FID Zero-shot FID


AttnGAN (Xu et al., 2017) 35.49
DM-GAN (Zhu et al., 2019) 32.64
DF-GAN (Tao et al., 2020) 21.42
DM-GAN + CL (Ye et al., 2021) 20.79
XMC-GAN (Zhang et al., 2021) 9.33
LAFITE (Zhou et al., 2021) **8.12**
DALL-E (Ramesh et al., 2021) _∼_ 28
LAFITE (Zhou et al., 2021) 26.94
GLIDE (Nichol et al., 2021) **12.24**


Unconditional CM3-Medium 40.65
Unconditional CM3-Medium 36.51
Conditional CM3-Medium 36.78
Conditional CM3-Large 29.56


Table 2: We compare FID on MS-COCO 256 _×_ 256. Following Nichol et al. (2021) we sample
roughly 30k conditioned samples for our models, and compare against the entire validation set. We
use a temperature of 0.85 for both CM3 models. We use the implementation available from Seitzer
(2020).


to the loss of texture from representing images through discrete tokens (e.g., the text of the train
station is blurred, as is the text on the bus).


8


the main
entrance of the
U.S. Department
of State in
Washington, D.C.


a pickup truck
parked in a
layby on a
highway.



the white marble
exterior
standing atop of
its fac¸ade.


a large bus
parked in a
layby



outside of a
train station
building from
across the
street.


a tall red bus
is coming down
some tracks


a man standing
next to a horse
on a beach


a jet airliner
flying with a
cloudy sky in
the background.



a man posing for a man next to a

a photo. large horse.



a U.S. Air Force
B-52H
Stratofortress
on the flight
line at
Barksdale Air
Force Base,
Louisiana



the Austrian
Airbus A321
aircraft with
its Austrian
registration



Source Image Tokenized Image CM3-Caption-Beam CM3-Caption-CLIP Ground Truth


Figure 6: We provide qualitative samples for zero-shot image-captioning using the CM3-Large
model. Caption-Beam refers to generating caption using beam over prompts, while Caption-CLIP
uses CLIP to get the top-ranked caption from a 128 candidate set (64 from masked prompt, 64 from
causal prompt).


Quantitatively we measure the quality of CM3 zero-shot captioning by evaluating using BERTScore [2] (Zhang et al., 2019) with the RoBERTa-Large models (Liu et al., 2019) on the validation
set from MS-COCO. We opt for the use of semantic evaluation versus classical metrics such as
BLEU/METEOR because we notice that the vocabulary and sentence structure of zero-shot captioning with CM3 is not compatible with MS-COCO ground truth labels, although the generated
content is semantically similar. We present our quantitative result in Table 3. CM3-Large is capable
of achieving reasonable zero-shot captioning performance on the MS-COCO dataset.


Precision Recall F1


CM3-Caption-Beam 0.781 0.789 0.785
CM3-Caption-CLIP 0.863 0.866 0.864


Table 3: BERTScore numbers for zero-shot captioning with CM3.


[2We use the open-source BERTScore at: https://github.com/Tiiiger/bert_score. The eval-](https://github.com/Tiiiger/bert_score)
uation method is: roberta-large ~~L~~ 17 ~~n~~ o-idf ~~v~~ ersion=0.3.11(hug ~~t~~ rans=4.11.3) ~~f~~ ast-tokenizer


9


4.3 TEXT MODALITY


CM3 is not only a cross-modal model but is fully capable of acting as a stand-alone language model.
This is even reflected in our data, where we do not enforce every document to have images; therefore,
pure language modeling will also occur during training. We evaluate our CM3 models on a wide set
of varying language tasks.


4.3.1 ENTITY DISAMBIGUATION


We reproduce the evaluation setting described by De Cao et al. (2020) and Le & Titov (2018) using
the same candidate sets, datasets and evaluating using the InKB micro-F1 metric.


We aim to find a prompt capable of representing the more general end-to-end entity linking task in
the CM3 model. From there, a proper sequence scoring of the candidate set will provide us with
an approach to zero-shot entity disambiguation. Luckily HTML based Wikipedia contains very
rich annotations. Specifically below, we show an example of naturally occurring entity linking that
would occur in our Wikipedia subset of CM3 training data.


**Original:** _Manetho_ writes that these kings ruled from
<a title="Memphis, Egypt"> _Memphis_ </a>


**Prompt:** _Manetho_ writes that these kings ruled from <a
title="<mask:0>"> _Memphis_ </a>...<mask:0>


**Target:** _Manetho_ writes that these kings ruled from <a
title="<mask:0>"> _Memphis_ </a>...<mask:0> Memphis, Egypt


Using our scoring approach we can simply score the **Target** while swapping out the postfix after
<mask:0>.


In-domain Out-of-domain









|Model Type Method AIDA MSNBC AQUAINT ACE2004 CWEB WIKI*|Avg.|
|---|---|
|_Direct Supervision_<br><br><br><br><br><br><br><br>Ganea & Hofmann (2017)<br>92.2<br>93.7<br>88.5<br>88.5<br>77.9<br>77.5<br>Guo & Barbosa (2018)<br>89<br>92<br>87<br>88<br>77<br>84.5<br>Yang et al. (2018)<br>**95.9**<br>92.6<br>89.9<br>88.5<br>**81.8**<br>79.2<br>Shahbazi et al. (2019)<br>93.5<br>92.3<br>90.1<br>88.7<br>78.4<br>79.8<br>Yang et al. (2019)<br>93.7<br>93.8<br>88.2<br>90.1<br>75.6<br>78.8<br>Le & Titov (2019)<br>89.6<br>92.2<br>90.7<br>88.1<br>78.2<br>81.7<br>Fang et al. (2019)<br>94.3<br>92.8<br>87.5<br>91.2<br>78.5<br>82.8<br>**De Cao et al. (2020)**<br>93.3<br>94.3<br>89.9<br>90.1<br>77.3<br>**87.4**|86.4<br>86.2<br>88.0<br>87.1<br>86.7<br>86.8<br>87.9<br>88.8|
|_Direct Supervision_<br>_{_<br>CM3-Medium<br>93.5<br>94.2<br>90.1<br>90.4<br>76.5<br>86.9<br>CM3-Large<br>94.8<br>**94.8**<br>**91.1**<br>**91.4**<br>78.4<br>**88.7**|88.6<br>**89.8**|
|_Self Supervision (0-Shot){_<br>CM3-Medium<br>78.0<br>80.1<br>75.4<br>81.4<br>68.5<br>76.2<br>CM3-Large<br>80.1<br>80.8<br>77.7<br>82.8<br>72.4<br>80.2|76.6<br>79.0|


Table 4: Aligned with GENRE’s evaluation, we use Micro _F_ 1 (InKB) for the named entity disambiguation task. **Bold** indicates best model. We note that although *WIKI can be thought of as
being out-of-domain, given that English Wikipedia was used to pre-train CM3, it can be considered
in-domain as well.


As an additional datapoint for the representations learned from CM3 we completely replicate the
training and evaluation for the GENRE model (De Cao et al., 2020). [3] . Specifically we first finetune CM3 on the BLINK data (Wu et al., 2019). For the in-domain scenario, we fine-tune CM3 on
the AIDA-CoNLL dataset (Hoffart et al., 2011). We evaluate on the AIDA-CoNLL dataset for the
in-domain scenario and the MSNBC, AQUAINT, ACE2004, WNED-CWEB (CWEB) and WNEDWIKI (WIKI) for the out-of-domain scenario (De Cao et al., 2020; Guo & Barbosa, 2018). We
present our results in Figure 4.


Given the strong supervision naturally available in Wikipedia HTML, it is unsurprising that CM3
shows strong, non-trivial zero-shot performance on the named entity disambiguation across a wide
array of named entity disambiguation tasks.


[3https://github.com/facebookresearch/GENRE](https://github.com/facebookresearch/GENRE)


10


Furthermore, the fine-tuned HTLM-Large model outperforms previous entity linking specific models to achieve a new SOTA over the benchmarked datasets.


4.3.2 ENTITY LINKING


We next consider the more general entity linking task. We experiment with two settings zero-shot
assuming we know the location of the entities and the full fine-tuning setting following the exact
methodology proposed in De Cao et al. (2020). Specifically for the end-to-end Entity Linking, we
aim to reproduce the setting of Kolitsas et al. (2018). We evaluate using the aforementioned _InKB_
micro- _F_ 1 with the same defined in-domain and out-of-domain datasets as described by De Cao et al.
(2020). We use the exact same _in-domain_ and _out-of-domain_ datasets as well as evaluating the _InKB_
micro- _F_ 1 on the GERBIL benchmark platform (R¨oder et al., 2018). Furthermore, we use the same
decoding strategy for the zero-shot case by limiting the generative tokens to only available candidate
entities. Please refer to De Cao et al. (2020) for the full fine-tuning setup.


For both setting we evaluate on seven test sets: MSNBC, Derczynski (Der) (Derczynski et al., 2015),
KORE 50 (K50) (Hoffart et al., 2012), N3-Reuters-128 (R128), N3-RSS-500 (R500) (R¨oder et al.,
2014), and OKE challenge 2015 and 2016 (OKE15 and OKE16) (Nuzzolese et al., 2015).


In-domain Out-of-domain









|Method AIDA MSNBC Der K50 R128 R500 OKE15* OKE16*|Avg.|
|---|---|
|_Direct Supervision_<br><br><br><br><br><br><br><br>Hoffart et al. (2011)<br>72.8<br>65.1<br>32.6<br>55.4<br>46.4<br>**42.4**<br>**63.1**<br>0.0<br>Steinmetz & Sack (2013)<br>42.3<br>30.9<br>26.5<br>46.8<br>18.1<br>20.5<br>46.2<br>46.4<br>Moro et al. (2014)<br>48.5<br>39.7<br>29.8<br>55.9<br>23.0<br>29.1<br>41.9<br>37.7<br>Kolitsas et al. (2018)<br>82.4<br>72.4<br>34.1<br>35.2<br>**50.3**<br>38.2<br>61.9<br>52.7<br>Broscheit (2020)<br>79.3<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>Martins et al. (2019)<br>81.9<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>van Hulst et al. (2020)_†_<br>80.5<br>72.4<br>41.1<br>50.7<br>49.9<br>35.0<br>**63.1**<br>**58.3**<br>De Cao et al. (2020)<br>**83.7**<br>73.7<br>**54.1**<br>60.7<br>46.7<br>40.3<br>56.1<br>50.0|47.2<br>34.7<br>38.2<br>53.4<br>56.4<br>**58.2**|
|_Direct Supervision_<br>_{_<br>CM3-Medium<br>71.4<br>68.5<br>48.6<br>58.3<br>44.9<br>41.1<br>61.9<br>37.7<br>CM3-Large<br>79.9<br>**74.8**<br>53.2<br>**62.4**<br>47.1<br>**42.8**<br>61.9<br>52.7|54.1<br>**59.3**|
|_Self Supervision (0-Shot){_<br>CM3-Medium<br>20.4<br>18.6<br>20.1<br>35.1<br>30.6<br>32.1<br>36.6<br>0.0<br>CM3-Large<br>24.8<br>21.4<br>25.6<br>39.0<br>31.1<br>34.9<br>37.1<br>0.0|24.2<br>26.7|


Table 5: We report Micro _F_ 1 on our test sets for our entity linking task. **Bold** indicates best model.
Following De Cao et al. (2020) we use a _[†]_ to indicate results from the Wikipedia 2019 setting as
opposed to the 2014 setting (which has older dump and fewer entities).


We present our results in Table 5. We see that our CM3 are extremely competitive with entity-linking
specific models and that our CM3-Large model sets a new state-of-the-art. Furthermore, although
our zero-shot numbers are substantially worse, they are still non-trivial, implying that CM3 learns a
significant amount of implicit entity linking through our training setting.


4.3.3 SUMMARIZATION


We next look at CM3 performance on the zero-shot summarization task, specifically we replicate the
zero-shot evaluation methodology of Aghajanyan et al. (2021). For all summarization benchmarks,
we use ROUGE-1/2/L as our primary metrics to stay consistent with other literature (Lin, 2004). We
look at the same datasets as HTLM.


**Gigaword** consists of headlines from news articles (Napoles et al., 2012). The target summaries are
relatively short, consisting roughly on average of 10 BPE tokens.

**CNN/Dailymail** (Hermann et al., 2015) provides multi-sentence target summaries close to 3 sentences, or roughly 50 tokens.

**Reddit TIFU** (Kim et al., 2018) contains summaries of Reddit posts. Specifically, we use the _short_
subset of data . Compared to our other summarization datasets, this dataset is highly abstractive and
not based on news articles.

**XSum** (Narayan et al., 2018) provides abstractive single sentence summaries of news articles.


We utilize the same prompts as available in Aghajanyan et al. (2021). We use the same available
size hints but using the implicit size-hint methodology described in § 3.2. Most prompts follow the


11


same theme of infilling either a title element or an element describing a headline (either through
attributes or using the meta tag). For completeness, below is an example of a prompt that can do
basic summarization.


Model Gigaword CNN/DM Reddit TIFU XSum


PEGASUS-0S 23.39/07.59/20.20 32.90/13.28/29.38 14.66/3.06/10.17 19.27/3.00/12.72
HTLM-Auto-NS 27.56/10.17/24.57 33.40/13.45/30.10 06.71/1.98/07.86 15.15/2.54/10.91
HTLM-Auto-S 28.73/11.31/26.49 34.65/14.54/32.15 08.15/2.92/09.75 17.14/3.41/13.43
HTLM-Manual-S 31.61/10.80/28.60 38.51/16.10/33.89 **15.81/2.98/10.54** 22.34/4.12/14.56


CM3-M-Manual 29.15/09.70/27.87 37.16/14.75/31.42 09.56/2.65/07.48 20.14/3.15/13.89
CM3-L-Manual **32.12/10.95/28.78** **38.88/16.27/34.16** 12.14/2.12/07.98 **24.86/6.08/16.32**


Table 6: CM3 results on zero-shot summarization. HTLM-Manual denotes manually engineered
prompts with size hints, while HTLM-Auto-S and HTLM-Auto-NS indicate auto-prompting with
and without size hints, respectively. The metrics shown are ROUGE-1/ROUGE-2/ROUGE-L, respectively. CM3-Large sets a new state-of-the-art on three news-based summarization datasets.


We present our results in Table 6. Both CM3 models saw significantly less text than the HTLM
model, with 2.7TB of text. Furthermore, the prompts being used were tuned specifically for the
HTLM model and are being used with no changes for CM3. With these challenges, we still see that
CM3-Large sets new state-of-the-art zero-shot summarization for three datasets. We attribute the
performance degradation in Reddit-TIFU data to CM3 pre-training data only containing CC-NEWS
and Wikipedia, which will not contain the type of summarizations needed for Reddit-TIFU.


5 FINE-TUNING


We next want to measure the quality of internal representations for the end goal of fine-tuning. We
compare CM3 with a wide array of masked language model derived models such as T5 (Raffel et al.,
2019), RoBERTa (Liu et al., 2019), HTLM (Aghajanyan et al., 2021) tested on the standard GLUE
benchmark (Wang et al., 2018). For CM3 we look at three settings for fine-tuning; standard finetuning, better fine-tuning using adversarial methods (Aghajanyan et al., 2020), and better fine-tuning
over prompts derived from Aghajanyan et al. (2021). We delegate the specifics hyper-parameters of
the fine-tuning experiments to § A.3


MNLI QQP RTE QNLI MRPC CoLA SST-2 # Params
Acc-m/mm Acc Acc Acc Acc Mcc Acc


T5-Base 87.1/86.2 89.4 80.1 93.7 87.5 51.1 95.2 220M
RoBERTA 90.2/- 92.2 86.6 94.7 89.1 68.0 96.4 330M
RoBERTa-R3F 91.1/91.3 92.4 88.5 95.3 91.6 71.2 97.0 330M
BART-Large 89.9/90.1 92.5 87.0 94.9 90.4 62.8 96.6 400M
HTLM 90.3/91.4 92.6 87.1 95.1 90.8 64.3 96.9 400M
HTLM-R3F 91.4/92.1 92.8 89.1 95.4 91.5 69.4 97.1 400M
HTLM-R3F-Prompt 91.6/91.2 92.9 89.4 95.7 91.7 69.8 97.3 400M
T5-Large 89.9/89.6 89.9 87.2 94.8 89.9 61.2 96.3 770M
T5-3B 91.4/91.2 89.7 91.1 96.3 90.0 67.1 97.4 3B
T5-11B 92.2/91.9 90.6 92.8 96.9 90.4 71.6 97.5 11B


CM3-Medium 89.9/89.7 89.6 89.1 93.1 86.5 63.1 94.9 2.7B
CM3-Medium-Prompt 90.8/91.0 89.9 90.5 95.1 89.9 66.2 96.3 2.7B
CM3-Medium-RXF-Prompt 90.9/91.1 90.0 90.7 95.3 90.0 67.1 96.9 2.7B


CM3-Large 91.1/91.0 89.9 91.9 95.6 89.6 64.6 94.2 13B
CM3-Large-Prompt 91.5/91.4 90.1 92.4 96.2 90.1 70.9 97.1 13B
CM3-Large-RXF-Prompt 91.9/91.5 91.1 92.5 96.4 90.3 70.8 97.3 13B


Table 7: Results on the GLUE development set for various fine-tuning methods applied to CM3.


We present our results in Table 7. Overall we see that both CM3 models are competitive against
T5 given the same parameter setting. Furthermore, aligning with the results from Aghajanyan et al.


12


(2021) we see that placing the natural language utterances of the various GLUE tasks into an HTML
prompt while fine-tuning non-trivially improves end-finetuning performance. The following experiments show that the causally masked language modeling approach is not detrimental to learning
fine-tunable representations, and neither is jointly modeling image tokens.


6 ETHICAL CONSIDERATIONS


Prior work has explored the extent to which language models encode harmful gender and racial
biases that parallel humans through the Word Embedding Association Test (WEAT) (Caliskan
et al., 2017), the Sentence Encoder Association Test (SEAT) (May et al., 2019) and the GroundedWEAT/Grounded-SEAT (Ross et al., 2021) metrics for multimodal language models (Tan & Celis,
2019). Given the generative nature of CM3 in both the language and visual modalities, we used
GWEAT/GSEAT to probe our model. Overall, we evaluated six bias tests for gender and seven
bias tests for race and found that our family of CM3 models show significantly less bias than other
models, speicifically VisualBERT (Li et al., 2019) and ViLBert (Lu et al., 2019).


Level VisualBert ViLBert CM3-Medium CM3-Large


C6: M/W, Career/Family S 1.05 1.14 0.00 0.98
W 0.54 0.51 0.10 0.12


C8: Science/Arts, M/W S 0.86 1.05 -0.09 0.42
W 0.62 0.14 0.08 0.07


C11: M/W, Pleasant/Unpleasant S -0.74 -0.84 0.00 -0.64
W -0.66 -0.31 -0.20 -0.48


Double Bind: M/W, Competent S -0.10 -0.04 0.01 -0.01
W -0.23 0.30 -0.07 -0.27


Double Bind: M/W, Likeable S -0.11 -1.12 -0.24 -0.59
W -0.60 0.09 0.00 0.10


Occupations: M/W, Occupation S 0.98 1.82 0.03 0.62
W 0.91 1.80 0.00 0.58


Total Significant Bias Count    - 5 6 0 2


Table 8: Following Ross et al. (2021) we present the results for all gender bias classes on answering
the question: “Do joint embeddings contain biases”? The numbers in our table represent effect sizes,
and are underlined if their respective p-values are below 0.05. Each bias type and model are tested
three times against Word embeddings (W) and Sentence embeddings (S).


We present our empirical results for gender and race bias in Table 8 and Table 9 respectively. Overall,
both CM3 have significantly less bias than other competing models, most likely due to our choice
to use only Wikipedia and CC-NEWS articles as training sources (and recent CC-NEWS articles at
that). Furthermore, we believe the fact that CM3-Medium shows no to very little signs of bias can
be an indicator of under-fitting as the large model is, unfortunately, able to show some bias from our
training data.


We also qualitatively experiment with whether CM3 can be prompted to produce harmful or objectionable images. In general, we noticed it was incredibly hard to produce such content, additionally
the lack of the ability to generate distinctive features of VQVAE-GAN acts to our benefit in terms
of preserving privacy.


7 RELATED WORK


Fundamentally our work is an extension of the HTLM work proposed by Aghajanyan et al. (2021)
to using the newly proposed causally masked objective, integrating images through VQVAE-GAN
tokens, and scaling up over an order of magnitude. From there, the individual capabilities of our
models are comparable to individual approaches.


13


Level VisualBert ViLBert CM3-Medium CM3-Large


C3: EA/AA, Pleasant/Unpleasant W 0.23 0.14 -0.44 0.10
S 0.31 -0.14 -0.057 0.05


C12: EA/AA, Career/Family W -0.29 0.43 0.117 0..23
S -0.54 0.34 -0.049 0.28


C13: EA/AA, Science/Arts W 0.04 0.21 0.325 0.12
S 0.12 0.68 0.169 0.465


Double Bind: EA/AA, Competent W 0.61 0.87 -0.535 0.42
S 0.24 0.25 0.0 0.18


Double Bind: EA/AA, Likeable W 0.21 -0.23 -0.535 0.19
S 0.27 -0.74 -0.535 0.21


Occupations: EA/AA, Occupation W -0.40 0.02 -0.51 0.01
S -0.41 0.46 -0.17 0.38


Angry Black Woman Stereotype W -0.07 0.26 -1.89 0.21
S -0.50 0.47 0.0 -0.10


Total Significant Bias Count   - 4 5 1 3


Table 9: Following Ross et al. (2021) we present the results for all racial bias classes on answering
the question: “Do joint embeddings contain biases”? Our table uses the same annotations as Table 8.


For example, the conditional and unconditional image generation capabilities of our model are most
similar in approach to DALL-E, which trains a left-to-right causal model over the concatenation of
textual tokens and VQ-VAE visual tokens (Ramesh et al., 2021). At the same time, the use of autoregressive modeling in entity linking and disambiguation was proposed by the GENRE in De Cao
et al. (2020).


The method of tokenizing non-discrete modalities to use standard sequence modeling approaches
have been extensively explored with DALL-E for images, Jukebox for Music (Dhariwal et al., 2020)
and vq-wav2vec for Speech (Baevski et al., 2019).


8 CONCLUSION


In this paper, we present the CM3 model, a causally masked trained language model that is capable of
non-trivial zero-shot performance on a wide range of zero-shot uni- and cross-modal tasks. We first
describe a new sequence modeling objective we call causally masked, enabling both full generative
modeling with bidirectional context.


Through extensive experimentation, we show that as a single model CM3 can be prompted to recover the functionality of many other models being able to do image generation, image captioning,
unconditional image generation, and more. Empirically we improve over state-of-the-art zero-shot
summarization, entity linking, entity disambiguation, highlighting the structure from the hypertext
during training. We show that representations learned by CM3 are not only useful for zero-shot
prompting but for fine-tuning by fine-tuning CM3 and state-of-the-art for entity linking and entity
disambiguation in general, all while staying highly competitive with T5 models on the GLUE benchmark.


REFERENCES


Armen Aghajanyan, Akshat Shrivastava, Anchit Gupta, Naman Goyal, Luke Zettlemoyer, and Sonal
Gupta. Better fine-tuning by reducing representational collapse. _arXiv preprint arXiv:2008.03156_,
2020.


Armen Aghajanyan, Dmytro Okhonko, Mike Lewis, Mandar Joshi, Hu Xu, Gargi Ghosh, and Luke
Zettlemoyer. Htlm: Hyper-text pre-training and prompting of language models. _arXiv preprint_
_arXiv:2107.06955_, 2021.


14


Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, et al. Efficient large scale language
modeling with mixtures of experts. _arXiv preprint arXiv:2112.10684_, 2021.


Alexei Baevski, Steffen Schneider, and Michael Auli. vq-wav2vec: Self-supervised learning of
discrete speech representations. _arXiv preprint arXiv:1910.05453_, 2019.


Mandeep Baines, Shruti Bhosale, Vittorio Caggiano, Naman Goyal, Siddharth Goyal, Myle Ott,
Benjamin Lefaudeux, Vitaliy Liptchinsky, Mike Rabbat, Sam Sheiffer, Anjali Sridhar, and Min
Xu. Fairscale: A general purpose modular pytorch library for high performance and large scale
[training. https://github.com/facebookresearch/fairscale, 2021.](https://github.com/facebookresearch/fairscale)


Abeba Birhane, Vinay Uday Prabhu, and Emmanuel Kahembwe. Multimodal datasets: misogyny,
pornography, and malignant stereotypes. _arXiv preprint arXiv:2110.01963_, 2021.


Samuel Broscheit. Investigating entity knowledge in bert with simple neural end-to-end entity linking. _arXiv preprint arXiv:2003.05473_, 2020.


Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. _arXiv preprint arXiv:2005.14165_, 2020.


Aylin Caliskan, Joanna J. Bryson, and Arvind Narayanan. Semantics derived automatically from
language corpora contain human-like biases. _Science_, 356:183 – 186, 2017.


Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. Autoregressive entity retrieval.
_arXiv preprint arXiv:2010.00904_, 2020.


Leon Derczynski, Diana Maynard, Giuseppe Rizzo, Marieke Van Erp, Genevieve Gorrell, Rapha¨el
Troncy, Johann Petrak, and Kalina Bontcheva. Analysis of named entity recognition and linking
for tweets. _Information Processing & Management_, 51(2):32–49, 2015.


Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_, 2018.


Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever.
Jukebox: A generative model for music. _arXiv preprint arXiv:2005.00341_, 2020.


Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image
synthesis. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recog-_
_nition_, pp. 12873–12883, 2021.


Alexander R Fabbri, Simeng Han, Haoyuan Li, Haoran Li, Marjan Ghazvininejad, Shafiq Joty,
Dragomir Radev, and Yashar Mehdad. Improving zero and few-shot abstractive summarization
with intermediate fine-tuning and data augmentation. _arXiv preprint arXiv:2010.12836_, 2020.


Zheng Fang, Yanan Cao, Qian Li, Dongjie Zhang, Zhenyu Zhang, and Yanbing Liu. Joint entity
linking with deep reinforcement learning. In _The World Wide Web Conference_, pp. 438–447,
2019.


Octavian-Eugen Ganea and Thomas Hofmann. Deep joint entity disambiguation with local neural
attention. _arXiv preprint arXiv:1704.04920_, 2017.


Zhaochen Guo and Denilson Barbosa. Robust named entity disambiguation with random walks.
_Semantic Web_, 9(4):459–479, 2018.


Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa
Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In _Advances in_
_neural information processing systems_, pp. 1693–1701, 2015.


Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.
Gans trained by a two time-scale update rule converge to a local nash equilibrium. _Advances in_
_neural information processing systems_, 30, 2017.


15


Johannes Hoffart, Mohamed Amir Yosef, Ilaria Bordino, Hagen F¨urstenau, Manfred Pinkal, Marc
Spaniol, Bilyana Taneva, Stefan Thater, and Gerhard Weikum. Robust disambiguation of named
entities in text. In _Proceedings of the 2011 Conference on Empirical Methods in Natural Lan-_
_guage Processing_, pp. 782–792, 2011.


Johannes Hoffart, Stephan Seufert, Dat Ba Nguyen, Martin Theobald, and Gerhard Weikum. Kore:
keyphrase overlap relatedness for entity disambiguation. In _Proceedings of the 21st ACM inter-_
_national conference on Information and knowledge management_, pp. 545–554, 2012.


Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language
models. _arXiv preprint arXiv:2001.08361_, 2020.


Byeongchang Kim, Hyunwoo Kim, and Gunhee Kim. Abstractive summarization of reddit posts
with multi-level memory networks. _arXiv preprint arXiv:1811.00783_, 2018.


Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint_
_arXiv:1412.6980_, 2014.


Nikolaos Kolitsas, Octavian-Eugen Ganea, and Thomas Hofmann. End-to-end neural entity linking.
_arXiv preprint arXiv:1808.07699_, 2018.


Phong Le and Ivan Titov. Improving entity linking by modeling latent relations between mentions.
_arXiv preprint arXiv:1804.10637_, 2018.


Phong Le and Ivan Titov. Boosting entity linking performance by leveraging unlabeled documents.
_arXiv preprint arXiv:1906.01250_, 2019.


Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer
Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pretraining for natural language generation, translation, and comprehension. _arXiv preprint_
_arXiv:1910.13461_, 2019.


Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple
and performant baseline for vision and language. _arXiv preprint arXiv:1908.03557_, 2019.


Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization_
_branches out_, pp. 74–81, 2004.


Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _European_
_conference on computer vision_, pp. 740–755. Springer, 2014.


Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining
approach. _arXiv preprint arXiv:1907.11692_, 2019.


Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. _arXiv preprint arXiv:1908.02265_, 2019.


Pedro Henrique Martins, Zita Marinho, and Andr´e FT Martins. Joint learning of named entity
recognition and entity linking. _arXiv preprint arXiv:1907.08243_, 2019.


Chandler May, Alex Wang, Shikha Bordia, Samuel R. Bowman, and Rachel Rudinger. On measuring social biases in sentence encoders. _ArXiv_, abs/1903.10561, 2019.


Andrea Moro, Alessandro Raganato, and Roberto Navigli. Entity linking meets word sense disambiguation: a unified approach. _Transactions of the Association for Computational Linguistics_, 2:
231–244, 2014.


Courtney Napoles, Matthew R Gormley, and Benjamin Van Durme. Annotated gigaword. In
_Proceedings of the Joint Workshop on Automatic Knowledge Base Construction and Web-scale_
_Knowledge Extraction (AKBC-WEKEX)_, pp. 95–100, 2012.


16


Shashi Narayan, Shay B Cohen, and Mirella Lapata. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. _arXiv preprint_
_arXiv:1808.08745_, 2018.


Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew,
Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with
text-guided diffusion models. _arXiv preprint arXiv:2112.10741_, 2021.


Andrea Giovanni Nuzzolese, Anna Lisa Gentile, Valentina Presutti, Aldo Gangemi, Dar´ıo
Garigliotti, and Roberto Navigli. Open knowledge extraction challenge. In _Semantic Web Evalu-_
_ation Challenges_, pp. 3–15. Springer, 2015.


Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier,
and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. _arXiv preprint_
_arXiv:1904.01038_, 2019.


Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style,
high-performance deep learning library. _Advances in neural information processing systems_, 32:
8026–8037, 2019.


Steven T Piantadosi. Zipf’s word frequency law in natural language: A critical review and future
directions. _Psychonomic bulletin & review_, 21(5):1112–1130, 2014.


Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language
models are unsupervised multitask learners. _OpenAI Blog_, 1(8):9, 2019.


Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. _arXiv preprint arXiv:2103.00020_, 2021.


Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text
transformer. _arXiv preprint arXiv:1910.10683_, 2019.


Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen,
and Ilya Sutskever. Zero-shot text-to-image generation. _arXiv preprint arXiv:2102.12092_, 2021.


Michael R¨oder, Ricardo Usbeck, Sebastian Hellmann, Daniel Gerber, and Andreas Both. N [3] -a
collection of datasets for named entity recognition and disambiguation in the nlp interchange
format. In _LREC_, pp. 3529–3533, 2014.


Michael R¨oder, Ricardo Usbeck, and Axel-Cyrille Ngonga Ngomo. Gerbil–benchmarking named
entity recognition and linking consistently. _Semantic Web_, 9(5):605–625, 2018.


Candace Ross, Boris Katz, and Andrei Barbu. Measuring social biases in grounded vision and
language embeddings. _ArXiv_, abs/2002.08911, 2021.


[Maximilian Seitzer. pytorch-fid: FID Score for PyTorch. https://github.com/mseitzer/](https://github.com/mseitzer/pytorch-fid)
[pytorch-fid, August 2020. Version 0.2.1.](https://github.com/mseitzer/pytorch-fid)


Hamed Shahbazi, Xiaoli Z Fern, Reza Ghaeini, Rasha Obeidat, and Prasad Tadepalli. Entityaware elmo: Learning contextual entity representation for entity disambiguation. _arXiv preprint_
_arXiv:1908.05762_, 2019.


Nadine Steinmetz and Harald Sack. Semantic multimedia information retrieval based on contextual
descriptions. In _Extended Semantic Web Conference_, pp. 382–396. Springer, 2013.


Yi Chern Tan and L. Elisa Celis. Assessing social and intersectional biases in contextualized word
representations. In _NeurIPS_, 2019.


Ming Tao, Hao Tang, Songsong Wu, Nicu Sebe, Xiao-Yuan Jing, Fei Wu, and Bingkun Bao. Dfgan: Deep fusion generative adversarial networks for text-to-image synthesis. _[arXiv:2008.05865](https://arxiv.org/abs/2008.05865)_,
2020.


17


Johannes M van Hulst, Faegheh Hasibi, Koen Dercksen, Krisztian Balog, and Arjen P de Vries. Rel:
An entity linker standing on the shoulders of giants. In _Proceedings of the 43rd International ACM_
_SIGIR Conference on Research and Development in Information Retrieval_, pp. 2197–2200, 2020.


Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE:
A multi-task benchmark and analysis platform for natural language understanding. In _Proceed-_
_ings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks_
_for NLP_, pp. 353–355, Brussels, Belgium, November 2018. Association for Computational Lin[guistics. doi: 10.18653/v1/W18-5446. URL https://www.aclweb.org/anthology/](https://www.aclweb.org/anthology/W18-5446)
[W18-5446.](https://www.aclweb.org/anthology/W18-5446)


Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke Zettlemoyer. Scalable zeroshot entity linking with dense entity retrieval. _arXiv preprint arXiv:1911.03814_, 2019.


Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong
He. Attngan: Fine-grained text to image generation with attentional generative adversarial networks. _[arXiv:1711.10485](https://arxiv.org/abs/1711.10485)_, 2017.


Xiyuan Yang, Xiaotao Gu, Sheng Lin, Siliang Tang, Yueting Zhuang, Fei Wu, Zhigang Chen, Guoping Hu, and Xiang Ren. Learning dynamic context augmentation for global entity linking. _arXiv_
_preprint arXiv:1909.02117_, 2019.


Yi Yang, Ozan Irsoy, and Kazi Shefaet Rahman. Collective entity disambiguation with structured
gradient tree boosting. _arXiv preprint arXiv:1802.10229_, 2018.


Hui Ye, Xiulong Yang, Martin Takac, Rajshekhar Sunderraman, and Shihao Ji. Improving text-toimage synthesis using contrastive learning. _[arXiv:2107.02423](https://arxiv.org/abs/2107.02423)_, 2021.


Han Zhang, Jing Yu Koh, Jason Baldridge, Honglak Lee, and Yinfei Yang. Cross-modal contrastive
learning for text-to-image generation. _[arXiv:2101.04702](https://arxiv.org/abs/2101.04702)_, 2021.


Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. _arXiv preprint arXiv:1904.09675_, 2019.


Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu,
Jinhui Xu, and Tong Sun. Lafite: Towards language-free training for text-to-image generation.
_[arXiv:2111.13792](https://arxiv.org/abs/2111.13792)_, 2021.


Minfeng Zhu, Pingbo Pan, Wei Chen, and Yi Yang. Dm-gan: Dynamic memory generative adversarial networks for text-to-image synthesis. _[arXiv:1904.01310](https://arxiv.org/abs/1904.01310)_, 2019.


18


A APPENDIX


A.1 MODEL ARCHITECTURE


For model architecture we use the same exact architecture for CM3-Medium and CM3-Large as the
dense 2.7B and 13B models described in Artetxe et al. (2021).


CM3-Large CM3-Medium


–decoder-embed-dim 5120 2560
–decoder-output-dim 5120 2560
–decoder-input-dim 5120 2560
–decoder-ffn-embed-dim 20480 10240
–decoder-layers 40 32
–decoder-normalize-before True True
–decoder-attention-heads 40 32
–share-decoder-input-output-embed True True
–decoder-learned-pos False False


Table 10: FairSeq architecture designation for CM3 models


A.2 UNIFORMITY OF VQVAE-GAN TOKENS


We plot a histogram of all image tokens in a subset of our data spanning 100k tokens. We see a
somewhat clear uniformity in tokens used.


Figure 7: Histogram of VQ-VAE-GAN Tokens in the CM3 Training Dataset.


A.3 FINETUNING GLUE HYPER-PARAMETERS


For our fine-tuning GLUE related experiments with the RXF method we use the following hyperparameters.


Hyper Parameter MNLI QNLI QQP SST-2 RTE MRPC CoLA


Learning Rate 5e-6 5e-6 5e-6 5e-6 1e-5 1e-5 1e-5
Max Updates 123873 33112 113272 20935 3120 2296 5336
Max Sentences 8 8 32 32 8 16 16


Table 11: Task specific hyper parameters for GLUE experiments


19


Hyper parameter Value


Optimizer Adam
Adam-betas (0.9, 0.98)
Adam-eps 1e-6
LR Scheduler polynomial decay
Dropout 0.1
Weight Decay 0.01
Warmup Updates 0.06 * max updates



Hyper parameter Value


_λ_ [0.1, 0.5, 1.0, 5.0]
Noise Types [ _U_, _N_ ]
_σ_ 1 _e −_ 5



Table 12: Hyper parameters for fine-tuning experiments on GLUE


20


