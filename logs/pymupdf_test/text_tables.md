## CM3: A CAUSAL MASKED MULTIMODAL MODEL OF
### THE INTERNET


|Col1|Armen|Aghajanyan, Bern|ie Huang∗, C|anda|ce Ross∗, Vla|d Karpukhi|n∗, Hu Xu∗, Naman|Goyal,|
|---|---|---|---|---|---|---|---|---|
||**Dmytr**|**o Okhonko, Manda**|**  r Joshi, Gar**|**    gi Gh**|**     osh, Mike Le**|**       wis, Luke Z**|**         ettlemoyer**||
||Facebo|ok AI Research|||||||
||_{_arme|nag,berniehua|ng,ccross|,vla|dk,huxu,n|aman_}_@fb.|com||
||_{_oxo,|mandarj,gghos|h,mikelew|is,l|sz_}_@fb.co|m|||
||||||||||
||||||||||
|22||||ABS|TRACT||||
|0|||||||||
|2||We introduce CM3,|a family of c|ausally|masked gen|erative mode|ls trained over a||
|||large corpus of stru|ctured multi-|odal|documents t|at can cont|in both text and||
||||||||||
|an||image tokens. Our|new causally|mask|ed approach|generates tok|ens left to right||
|J||while also masking|out a small n|umbe|r of long toke|n spans that|are generated at||
||||||||||
|9||the end of the strin|, instead of|heir o|iginal positi|ns. The cas|al masking ob-||
|1||ject provides a type<br>|of hybrid of<br>|the m<br>|ore common<br>|causal and m<br>|asked language<br>||
|||models, by enablin|g full generat|ive m|odeling while|also providi|ng bidirectional||
|]||context when gener|ating the mas|ked sp|ans. We trai|n causally m|asked language-||
|L||image models on la|rge-scale we|b and|Wikipedia ar|ticles, where|each document||
|C||contains all of the t|ext, hypertex|t mark|up, hyperlin|ks, and imag|e tokens (from a||
|s.||VQVAE-GAN), pro|vided in the|order|they appear i|n the origina|l HTML source||
|c||(before masking). T|he resulting|CM3 m|odels can ge|nerate rich s|tructured, multi-||
|[||modal outputs whil|e conditionin|g on|arbitrary mas|ked docume|nt contexts, and||
|||thereby implicitly le|arn a wide ra|nge o|f text, image,|and cross m|odal tasks. They||
|1||<br>can be prompted to|<br>    recover, in a|<br>       zero|<br>       -shot fashion|<br>        , the functio|<br>          nality of models||
|0v||such as DALL-E, G|ENRE, and H|TLM|(Ramesh et a|l., 2021; De|Cao et al., 2020;||
|2||Aghajanyan et al.,|2021). We s|et the|new state-of|-the-art in ze|ro-shot summa-||
|5||rization, entity linki|ng, and entit|y disa|mbiguation w|hile maintain|ing competitive||
|7||performance in the|ﬁne-tuning se|tting.|We can gen|erate images|unconditionally,||
|0||conditioned on text|(like DALL|-E) an|d do caption|ing all in a z|ero-shot setting||
|.||with a single model|.||||||
|201||<br>|||||||
|2|1<br>INT|RODUCTION|||||||
|<br>:|||||||||
|<br>iv|Recent a|dvancements in larg|e-scale genera|tive se|quence mod|eling have sig|niﬁcantly improved|zero-|
|<br>X|shot perf|ormance on multipl|e modalities,|inclu|ding text Bro|wn et al. (20|20); Fabbri et al. (2|020);|
|<br>r|Aghajan|yan et al. (2021) an|d images Ra|mesh e|t al. (2021).|Recent wor|k has also shown h|ow to|
|<br>a|use docu|ment structure, e.g.,|as provided b|y HT|ML web mark|up, to enable|more effective zero|-shot|
||promptin|g for text-only tasks|(Aghajanyan|et al.,|2021). In thi|s paper, we s|how it is possible to|learn|
||multi-mo|dal document-struc|tured generat|ive m|odels, to join|tly represent|formatted hypertex|t and|
||images a|s they naturally co-o|ccur within f|ull do|cument conte|xts.|||
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|||
||We intro|duce CM3, a family|of causally|maske|d generative|models train|ed over a large corp|us of|
||structure|d multi-modal docu|ments. Causa|lly m|asked models|generate tok|ens left to right, jus|t like|
||a causal|language model, bu|t also mask|out a s|mall number|of long tok|en spans, which are|then|
||generate|d at the end of the s|tring instead|of thei|r original po|sitions. This|provides a new hyb|rid of|
||causal an|d masked language|models, enab|ling f|ull generativ|e modeling w|ith bidirectional co|ntext.|
||For exam|ple, it can also be|used in our s|etting|to inﬁll com|plete images|or larger structure|d text|
||sections,|conditioned on the|rest of the do|cumen|t.||||
|<br>|<br>|<br>|<br>|<br>|<br>||||
||We train|CM3 models on clo|se to a teraby|te of w|eb-based da|ta following|Aghajanyan et al. (2|021),|
||extended|to include images t|hrough VQV|AE-GA|N tokens (E|sser et al., 20|21) and additional h|yper-|
||text link|structure. This data|is in strong c|ontras|t to previous|methods that|were either uni-mo|dal or|
||<br>|<br>|<br>||||||
||_∗_Equa|l Contribution for Sec|ond Author||||||
||||||||||
||||||1||||


|carefully curate|d the im|age-text alignment (e|.g., for image caption|ing Radford et al.|(2021); Ramesh|
|---|---|---|---|---|---|
|et al. (2021)).|We trai|n a 2.7 billion and 13|billion causally mas|ked model on thi|s data which we|
|call CM3-Mediu|m and|CM3-Large respective|ly.|||
|<br>|<br>|<br>|<br>|||
|Extensive exper|iments|demonstrate that thes|e models are able to|perform a wide ra|nge of zero-shot|
|uni- and cross-m|odal ta|sks. We show both qu|alitatively and quanti|tatively that CM3|can be prompted|
|for non-trivial i|mage g|eneration, similar to t|hat of DALL-E. We a|lso show that CM|3 models are ca-|
|pable of improv|ing ove|r state-of-the-art zero|-shot summarization,|entity linking, en|tity disambigua-|
|tion, highlightin|g the s|tructure that comes fr|om the hypertext duri|ng training. Fina|lly, we show that|
|by ﬁne-tuning|CM3 we|set the new state-of-|the-art for entity lin|king and entity d|isambiguation in|
|general.||||||
|||||||
|To summarize,|our con|tributions include:||||
|<br>|<br>|<br>||||
|• We pre|sent th|e ﬁrst hyper-text lang|uage-image model,|trained on close|to a Terabyte of|
|multi-|modal s|impliﬁed HTML data|from the common cr|awl.||
|<br>|<br>|<br>|<br>|<br>||
|• We pre|sent th|e causally masked obj|ective, a hybrid of ca|usal and masked|language models|
|that all|ows fo|r bidirectional context|control during gener|ative mask inﬁlli|ng.|
|<br>|<br>|<br>|<br>|<br>|<br>|
|• We de|monstr|ate consistently strong|transfer from CM3 t|o a range of uni-|modal and multi-|
|modal|tasks a|t differing supervision|levels, including sta|ting state-of-the-|art on entity dis-|
|ambig|uation a|nd zero-shot summari|zation.|||
|<br>|<br>|<br>|<br>|||
|• We rel|ease all|code and models to s|upport future CM3 re|search.||
|<br><br>|<br>|<br>|<br>|||
|2<br>CAUSAL|LY M|ASKED OBJECTIV|E|||
||<br>|<br>||||
|Traditional app|roache|s to pre-training hav|e focused on mixin|g the architectur|al choices (i.e.,|
|encoder-only, d|ecoder-|only, encoder-decoder|) with objective choi|ces (i.e., masking,|causal language|
|modeling). For|exam|ple, masked encoder-|only models such as|BERT (Devlin|et al., 2018) and|
|RoBERTa (Liu|et al.,|2019) excel in non-g|enerative ﬁne-tuning|tasks. Masked|encoder-decoder|
|models such as|BART|(Lewis et al., 2019) a|nd T5 (Raffel et al.,|2019) excel in bo|th discriminative|
|and generative ﬁ|ne-tun|ing. Brown et al. (202|0) on the other hand,|showed that causa|l language mod-|
|els (de-facto, de|coder|only) are capable of n|on-trivial performan|ce without the ne|ed of ﬁne-tuning|
|by simply prom|pting w|ith appropriate string|to control the genera|ted outputs Radf|ord et al. (2019);|
|Brown et al. (20|20); A|rtetxe et al. (2021).||||
|<br>|<br>|<br>||||
|There are pros a|nd con|s to both masked and|causal language mod|eling in the conte|xt of prompting.|
|Masking offers|the crit|ical ability to encode|bi-directionality with|in the prompts at|the cost of only|
|decoding rough|ly 15%|of the tokens of the|input sequence dur|ing training (Dev|lin et al., 2018;|
|Liu et al., 2019;|Lewis|et al., 2019). Convers|ely, decoder-only cau|sal language mod|els decode every|
|token in the inp|ut sequ|ence in the training but|are typically limited|to left-only conte|xts. Empirically,|
|more work has|also be|en done on scaling cau|sal decoder-only rat|her than their cou|nterparts.|
|<br>|<br>|<br>|<br>|<br>|<br>|
|In an effort to|get mo|st of the best of both|worlds, we introduc|e a novel objectiv|e that combines|
|the beneﬁt of pe|r-token|generation with optio|nal bi-directionality|speciﬁcally tailor|ed to prompting.|
|For a document|of size|_ s_ we select_ n ∼_Cla|mp(Poisson(1)_,_ 1|_,_ 16) masks and|for each of those|
|masks we selec|t span|_ m ∼_(_Uniform_(0_, s_|)_, Uniform_(0_, s_))|which does not in|tersect with any|
|other_ m_. These|values|are chosen to, on ave|rage, select relatively|few relatively lo|ng spans, which|
|we expect will a|llow th|e model to learn to in|ﬁll long spans. We th|en order these m|asks by the order|
|that they appear|in the|source document, rep|lace the span of the|mask in the sourc|e document with|
|an enumerated|mask to|ken (i.e., <mask:0>|, <mask:1>), and|move the masked|spans to the end|
|of the documen|t follow|ed by a unique end of|document token.|||
|<br>|<br>|<br>||||
|Figure 1 shows|the co|mplete process.||||
|<br>|<br>|<br>||||
|We also augme|nt the|standard cross-entrop|y loss to weigh the l|oss of predicting|mask tokens as|
|0, as they are p|laced a|t random locations w|hich carry no inform|ation to the und|erlying sequence|
|modeling objec|tive.|||||
|<br>|<br>|||||
|The complete a|rray of|beneﬁts will become|more apparent when|designing prom|pts for uni/cross-|
|modal tasks in|§ 4. H|owever, at the core, t|he causally masked|objective can do|causal language|
|modeling while|option|ally allowing for bidir|ectionality when nee|ded.||


2


|Col1|Causally<br>Masked Monte|Melkonian was|a|left-wing <mask:0>|nationalist militant .|<mask:0> <a|href= Armenian|_nationalism >|
|---|---|---|---|---|---|---|---|---|
||**Language**<br>**Model**||||||||
||||||||||
|||||<a|href=<br>Armenian<br>_nation|alism<br>>|||
||||||||||
||Masked||||||||
||Monte<br><br>Language|Melkonian<br>was|a|left-wing|<mask>|nationali|st<br>militant<br>.||
||Model||||||||
||||||||||
||Monte<br><br>Language<br>Model|Melkonian<br>was|a|<a<br>left-wing|href=<br>Armenian<br>_nation|alism<br>><br>nationali|st<br>militant<br>.||
||||||||||
|F|igure 1: A vis|ual represent|ation|of various|language mode|ling objective|s as well as|our proposed|
|c|ausal language|modeling o|bject|ive with a s|ingle mask (_n_|= 1). Given|the left-to-ri|ght nature of|
|c|ausal language|models (bo|ttom|row) we w|ould not be abl|e to generate|the Wikiped|ia entity link|
|h|ighlighted in o|range.|||||||
|<br>|<br>||||||||
|3|CM3||||||||
||||||||||
|A|ghajanyan et a|l. (2021) use|d str|uctured doc|uments for text-|only pre-train|ing with stro|ng zero-shot|
|p|erformance.** C**|ausally-**M**a|sked|** M**ultimodal|** M**odeling (CM|3) extends th|is work by|modeling full|
|d|ocument struc|ture includin|g im|ages and h|ypertext links.|Furthermore,|we move a|way from the|
|B|ART-like obje|ctive of Ag|hajan|yan et al. (|2021) to use ou|r new causal|ly masked o|bjective with|
|d|ecoder-only m|odels.|||||||
|<br>|<br><br>||||||||
|3|.1<br>DATA||||||||
||||||||||
|F|ollowing Agh|ajanyan et al|. (20|21) we aim|to implement|a transform o|ver HTML d|ocuments to|
|e|xtract out to m|inimal-HTM|L, i.e|., the minim|al set of text th|at is semantic|ally relevant|for end tasks.|
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|B|irhane et al. (|2021) gave i|n-dep|th criticism|s of Common|Crawl based|multi-modal|datasets and|
|s|howed the exi|stence of hig|hly p|roblematic|examples (i.e.,|explicit imag|es and text|pairs of rape,|
|p|ornography, a|nd ethnic slu|rs).|Given these|severe ethical|concerns, we|opt-out of p|rocessing all|
|o|f Common Cr|awl and inst|ead o|pt into usin|g a subset of t|he Common|Crawl News|(CC-NEWS)|
|d|ataset and all o|f English W|ikipe|dia.|||||
|<br>|<br>|<br>|<br>|<br>|||||
|G|iven a valid H|TML DOM|1 per|document,|we run several|passes to str|ip down the|DOM to the|
|e|lements of ma|ximal seman|tic va|lue. We ﬁrs|t remove all ele|ments which|do not contai|n textual ele-|
|m|ents. We also|ﬁlter out all|_ head_|_ ers_,_ footers_,|_ copyrights_,_ for_|_ ms_,_ dialog bo_|_ xes_ and_ iFra_|_ mes_. We fold|
|c|onsecutive <d|iv> elemen|ts int|o a singular|<div> eleme|nt with merge|d attributes.|Furthermore|
|w|e strip all the|attributes fro|m ev|ery elemen|t which are not|derived from|structured gr|aphs such as|
|O|penGraph, Sc|hema and Tw|itter.||||||
|<br>|<br>|<br>|<br>||||||
|F|or every <im|g> tag in the|doc|ument with|a valid src at|tribute URL,|we downloa|d the image,|
|r|esize to 256x2|56 pixels wit|h ran|dom croppi|ng and then tok|enize it with|VQVAE-GA|N from Esser|
|e|t al. (2021). T|his amounts|to 25|6 tokens f|or every image.|We then ins|ert the string|value of the|
|t|okens joined w|ith a space b|ack i|nto the src|attribute.||||
|<br>|<br>|<br>|<br>|<br>|<br>||||
||We do not plac|e any restric|tions|on the num|ber of images|or their locat|ions. We pre|sent a set of|
|h|igh-level statis|tics in Table|1.||||||
|||<br>|<br>|<br>|<br>||<br>||
|||Documents|(Mil|lion)<br>Size|(GB)<br>Uniqu|e Images (Mi|llion)<br>Tok|ens (Billion)|
|||<br>||<br> <br>|<br> <br>||<br> <br>||
|C|C-NEWS|45||460|18||121||
|E|n-Wikipedia|16||383|7||102||
||||||||||
|T|otal|61||843|25||223||
||||||||||
|||Table 1:|Hig|h level stati|stics of the data|used to train|CM3.||
|||<br>|<br>|<br>|<br>|<br>|||
||1The DOM o|r Document O|bject|Model is an|interface that tre|ats an HTML|document as a|tree structure|
|w|herein each nod|e is an object|repres|enting a part|of the document|.|||


3


|For experimentat|ion, we create|two test sets|from each d|ata source|with 10,000|u|nique|documents|
|---|---|---|---|---|---|---|---|---|
|for each. We de-|duplicated our|test sets to en|sure no ove|rlap betw|een test and tr|a|in set|s to the best|
|of our abilities.|||||||||
|<br><br>|||||||||
|3.2<br>SIZE HINT|S||||||||
|<br>|||||||||
|Aghajanyan et al|. (2021) introd|uced the con|cept of size|hints wh|ich allows the||user t|o guide the|
|model during sa|mple generatio|n through tok|en conditio|ning. Spe|ciﬁcally, HTL||M ins|erts a prob-|
|abilistic estimate|of the size of|the mask as|a token pos|t the mas|k token (e.g.,||<mas|k>12 for a|
|probabilistic size|of 12). For C|M3, we notice|d that size-h|ints degra|ded not only|e|nd-pe|rplexity but|
|also the zero-sho|t performance|on a signiﬁca|nt set of eval|uation tes|ts.||||
|<br>|<br>|<br>|<br>|<br>|<br>||||
|We also note tha|t we can impli|citly give a s|ize hint duri|ng mask|generation for||a sing|le mask by|
|asking the mode|l to generate c|ausally max|~~ s~~equence|~~ l~~ength|- size~~ h~~|i|nt to|kens before|
|placing the secon|dary <mask:|0> token.|||||||
|<br><br>|||||||||
|3.3<br>TRAINING|||||||||
||||||||||
|We train 4 model|s; 125M, 800M|, 2.7B, and 1|3B paramete|rs. The pu|rpose of the t|w|o sm|aller models|
|was to establish|basic hyper-par|ameters that|are viable fo|r the caus|ally masked l|a|nguag|e modeling|
|objective and the|refore were un|der-trained.|However, al|l downstr|eam tasks will||be ev|aluated with|
|our 2.7B model (|CM3-Medium)|and our 13B|model (CM|3-Large).|HTLM-Medi|u|m wa|s trained on|
|240 V100 GPU f|or 28 days, wh|ile HTLM-L|arge was tra|ined on 3|84 A100 GPU||for 2|4 days. Our|
|implementation|was in PyTorch|(Paszke et a|l., 2019) usi|ng fairse|q (Ott et al., 2|0|19) a|nd fairscale|
|(Baines et al., 2|021). For ever|y model, our|per GPU b|atch size|was 8, with|a|maxi|mum token|
|sequence length|of 2048. We u|se the polyno|mial decay l|earning ra|te scheduler a|v|ailab|le in Paszke|
|et al. (2019) with|1500 warmup|updates. We|clipped the|gradient|norms to 1.0 a|n|d use|d the Adam|
||||||||||
|optimizer with_ β_|1 = 0_._9,_ β_2|= 0_._98 (Kin|ma & Ba,|2014).|e defer our||odel|architecture|
|description to § A|<br>  .1.||||||||
|<br><br>|<br>||||||||
|3.4<br>SCALING|LAWS||||||||
|<br>|<br>||||||||
|Our training sett|ing has a coup|le of new pa|rameters tha|t can im|pact the tradit|i|onal s|caling laws|
|of causal langua|ge models. Th|e multi-mod|al nature of|our prop|osed model b|r|eaks t|he standard|
|assumptions of t|oken distributi|onality. Tradi|tionally lan|guage tok|ens are said t|o|follo|w a Zipﬁan|
|distribution (Pian|tadosi, 2014),|while image to|kens are stri|ctly unifo|rm (see § A.2|).|Furth|ermore, the|
|unrestricted locat|ions of the im|ages and text|introduce u|npredictab|le complexity|.|Last|ly, although|
|we are still comp|uting the joint|probability of|the docume|nt, we do|so in a round|a|bout|way through|
|shufﬂing of the d|ocument via th|e causally ma|sked objecti|ve. These|fundamental d|i|fferen|ces warrant|
|a quick look into|the scaling law|s of CM3.|||||||
||<br>|<br>|||||||
|||Perplexity Bas|ed Scaling La|ws for CM3|||||
||||||||||
|||||||M<br>|odel<br>||
|||||||C<br>|M3~~-~~XSmal<br>|l|
|||||||C<br>|M3~~-~~Small<br>||
|100<br>ty||100||100||C<br>|3~~-~~Mediu<br>|m|
|exi||||||C|M3~~-~~Large||
|Perpl|||||||||
|Validation|||||||||
||||||||||
|0|1<br>2<br>3<br>|4<br><br>0<br>50<br>|00 100000 150000<br>|200000|0<br>1<br>2<br>||3<br><br>||
||Number of Documents|1e8<br><br> <br>|<br>Number of Updates||Training Time (S|ec|onds)<br>|1e6|
||<br>||<br>||<br>||||
|Figure 2: Basic|perplexity base|d scaling law|s for the pro|posed CM|3 objective an|d|train|ing set-up.|
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|We present the va|rious perplexit|y curves for t|he four mod|els of var|ying sizes we|tr|ained|. Given that|
|our models were|trained on var|ious hardwar|e set-ups, w|e normal|ize the trainin|g|time|by linearly|
|scaling each exp|eriment timing|to 256 GPU.|Most impo|rtantly, w|e see healthy|s|calin|g, similar to|


4


|Kaplan et al.|(2020) without any|pathological ca|ses, implying the|re is still a|good amou|nt of gains|
|---|---|---|---|---|---|---|
|to be achieve|d with further scali|ng. An in-depth|analysis of the sc|aling laws o|f the causa|lly masked|
|objective is o|utside this current|work’s scope and|will be consider|ed for futur|e work.||
|<br><br>|<br>|<br>|||||
|4<br>ZERO/|FEW-SHOT PRO|MPTING|||||
||<br>||||||
|4.1<br>IMAGE|MODALITY||||||
||<br>||||||
|Although we|do not train on pur|e image docume|nts, CM3 can still|operate ov|er image ta|sks. To do|
|so, we must|cleverly describe th|e task through a|textual prompt, us|ing the <i|mg> tag.||
|<br><br>|<br>|<br>|<br>||||
|4.1.1<br>UNC|ONDITIONAL IMAG|E GENERATIO|N||||
||<br>|<br>|||||
|To sample fr|om the distribution o|f images availab|le to CM3, we can|simply ask|the model|to produce|
|the next set o|f tokens after the fo|llowing prompt:|<img.||||
|<br>|<br>|<br>|<br>||||
|Interestingly|enough, CM3 prefe|rs to ﬁrst generat|e a short descript|ion of the i|mage throug|h the alt|
|attribute and|then generate the im|age tokens via t|he src attribute.|We can forc|e the model|to directly|
|generate ima|ge tokens without ﬁ|rst giving a desc|ription with the f|ollowing pr|ompt: <im|g src=".|
|We consider|both prompts to te|st unconditiona|l image generatio|n since we|do not co|ndition the|
|image genera|tion but rather the|model self-condi|tions.||||
|<br>|<br>|<br>|<br>||||
|We sample a|ccording to the dist|ribution of the m|odel without alte|ring the te|mperature.|We present|
|a sample of n|on-cherry picked e|xamples in Figu|re 3.||||
|<br>|||||||
|<img|||||||
|||<br>|<br>|<br>|||
||(a) A moun|tain of<br>(b) Spai|n Europa<br>(c) blog|TIGI Bed<br>|(d)<br>birthda|y<br>in-|
||olive trees|on the<br>Amenace|r Winter<br>Head|Tie<br>Dye<br>|vitation<br>pri|ntable|
||way to Cab|o de la|Spray H|air Spray<br>|christmas gi|ft for|
||Vela||Hairspra|y ml<br>|birthday|party|
||||||Printable Te|mplate|
||||||||
|<img|src="||||||
|<br>|<br>||||||
|Figure 3: Fo|ur samples for two|of the prompts w|e proposed for un|conditiona|l image gen|eration for|
|CM3-Large.|For the self-caption|ed images we pl|ace the respective|caption un|der the ima|ge. Results|
|were selecte|d at random, with no|cherry picking.|||||
|<br>|<br>|<br>|<br>||||
|The model is|more than capable|of generating co|herent images. W|e note that|via this pro|mpting, we|
|can recover t|he full functionality|of the DALL-E|model proposed|in Ramesh|et al. (2021|). Interest-|
|ingly enough|, we see qualitative|improvements|with allowing the|model to f|ree generat|e a caption|
|prior to gene|rating.||||||
|<br>|<br>||||||
|We continue|by doing an empiric|al study of the u|nconditional gene|ration of C|M3, by gene|rating 30k|
|samples with|out textual conditio|ning and calcul|ating the Fr´echet|Inception|Distance (FI|D, Heusel|
|et al. (2017)|) over MS-COCO,|following the m|ethodology prop|osed in Nic|hol et al. (|2021) (Lin|
|et al., 2014).|We present our res|ults in the uniﬁe|d table showing F|ID calculat|ions in Tabl|e 2. With-|
|out any textu|al conditioning and|without explici|tly optimizing fo|r either MS|-COCO or|generation|


5


|(unlike other be|nchmarks in th|e table) CM3 La|rge approaches th|e FID perform|ance of mo|dern Gen-|
|---|---|---|---|---|---|---|
|erative Adversa|rial Networks (|GAN).|||||
|<br><br>|<br>||||||
|4.1.2<br>IMAGE|IN-FILLING||||||
|<br>|<br>||||||
|Unlike DALL-|E, which lever|ages left-to-rig|ht language mode|ling objective|to model|language-|
|image tokens,|CM3 with the|proposed causa|lly masked langu|age modeling|makes it|possible to|
|condition conti|guous sections|of an image on|the surrounding c|ontext for ima|ge in-ﬁllin|g. Speciﬁ-|
|cally, CM3 can i|nﬁll images wi|th the following|prompt:||||
|||<br>|<br>||||
|**Infil**|**ling Prompt**|** :**<br><img src=|"_{_prefix_}_<mas|k:0>_{_postf|ix_}_"><mas|k:0>|
||||||||
|Using the same|decoding strat|egies described|in § 4.1.1 we gen|erate uncondi|tional inﬁl|led images|
|with only CM3-|Large and pres|ent qualitative r|esults in Figure 4|. Overall we|see that CM|3-Large is|
|capable of gene|rating semantic|ally coherent in|ﬁlls even without|grounding in|text.||
|<br><br>|<br>|<br>|||||
|4.2<br>TEXT-IM|AGE MODALIT|Y|||||
||<br>||||||
|4.2.1<br>CONDI|TIONAL IMAGE|IN-FILLING|||||
||<br>|<br>|||||
|Additionally, C|M3 can further p|erform image i|n-ﬁlling condition|on the additio|nal text co|ntext. This|
|can be achieved|by slightly aug|menting the pr|ompt as follows:||||
|<br>|<br>|<br>|<br>||||
|**Condi**|**tional Infi**|** lling Prompt**|** :**||||
|||<br>|<br>||||
|<img|alt="Photo:|_{_text_}_" sr|c="_{_prefix_}_<m|ask:0>_{_post|fix_}_"><m|ask:0>|
||||||||
|We show qualit|ative results in|Figure 4. Imme|diately we notice|the substantia|l improve|ment in the|
|generated imag|e when grounde|d in ground tru|th text vs. uncond|itional image-|inﬁlling.||
|<br><br>|<br>|<br>|||||
|4.2.2<br>CONDI|TIONAL IMAGE|GENERATION|||||
||<br>|<br>|||||
|We can do con|ditional text ge|neration using|CM3 similar to D|ALL-E by us|ing a prop|er prompt.|
|Speciﬁcally by|conditioning us|ing the alt att|ribute of the img|tag.|||
||||<br>|<br>|||
|**Condi**|**tional Gene**|** ration Promp**|** t:**<br><img alt=|"_{_prompt_}_|||
||||||||
|We present qua|litative conditi|onal image gen|eration results in|Figure 5. Sp|eciﬁcally,|we sample|
|32 images for e|very prompt gi|ven and re-rank|using CLIP to g|et the top-4 i|mages (Rad|ford et al.,|
|2021). Overall|we see that CM|3 can generate|recognizable ima|ges of the inp|ut text. The|re are still|
|failure cases, s|uch as the seco|nd image in the|second prompt,|where the mo|del easily g|enerates a|
|landscape but f|orgets to gener|ate the red car.|The third prompt|, CM3, is inca|pable of d|rawing the|
|face of a sheep|while getting th|e general body|and texture correc|t.|||
|<br>|<br>|<br>|<br>|<br>|||
|We note that CM|3 trains with an|order of magni|tude less unique i|mages than D|ALL-E, and|the subset|
|of images avail|able to CM3 are|the images ava|ilable in news and|Wikipedia ar|ticles; ther|efore, CM3|
|does not genera|te ﬁctional ima|ges well. That b|eing said, casting|a larger pool|for CLIP s|election by|
|randomly samp|ling a larger set|qualitatively ﬁ|xes some of these|issues.|||
|<br>|<br>|<br>|<br>|<br>|||
|For quantitative|analysis, we c|ompute FID on|MS-COCO follo|wing the meth|odology p|rovided by|
|Nichol et al. (2|021). Speciﬁcal|ly, we sample 3|0k samples condi|tioned on MS|-COCO ca|ptions. For|
|all models, we|use a temperatu|re of 0.85 and d|o straightforward|sampling.|||
|<br>|<br>|<br>|<br>|<br>|||
|We present our|FID results on|MS-COCO 256|x256 in Table 2.|In general CM|3 is capabl|e of gener-|
|ating semantica|lly coherent im|ages on-par wit|h modern GANs.|Furthermore,|our conditi|onal CM3-|
|Large model ap|proaches the pe|rformance of th|e DALL-E model|while using a|n order of|magnitude|
|fewer data.|||||||
|<br><br>|||||||
|4.2.3<br>CAPTI|ONING||||||
||||||||
|We next look at|the dual-task to|conditional im|age generation an|d image capti|oning. We c|an prompt|
|CM3 to do zero|-shot image ca|ptioning by ask|ing the model to|generate eithe|r the alt|or title|


6


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|group of peo|ple|
|---|---|---|---|---|---|---|---|---|
||||||||windsurfing|over|
||||||||the beach an|d|
||||||||water in the||
||||||||ocean.||
||||||||||
||||||||the wooden p|ark|
||||||||benches are||
||||||||painted dark||
||||||||purple.||
||||||||||
||||||||some bread i|s on|
||||||||a plate with||
||||||||jam, an appl|e,|
||||||||yogurt and||
||||||||orange juice|.|
|||||||<br>|<br>|<br>|
||||||||a nice looki|ng|
||||||||hotel room w|ith|
||||||||a neatly don|e|
||||||||bed, coffee||
||||||||table, and a||
||||||||chair.||
||<br>|||<br>|||||
|Source Im|age<br>M|as|ked/Tokenize|d Image<br>CM3-Inﬁll|ing-U<br>CM3-Inﬁllin|g-C|Ground Tr|uth|
|<br>|<br> <br>|<br>|<br>|<br><br>|||<br>|<br>|
|Figure 4: W|e provide|q|ualitative sa|mples for zero-shot|image-inﬁlling using|the CM|3-Large mo|del|
|using the afo|rementio|ne|d prompts.|CM3-Inﬁlling-U ref|ers to inﬁlling witho|ut cond|itioning on t|ext|
|while CM3-I|nﬁlling-C|r|efers to con|ditioning on the grou|nd truth text.||||
|<br>|<br>|<br>|<br>|<br>|<br>||||
|attributes of|a properl|y|set <img>|tag. Due to attribute|s always appearing i|n alpha|betical order|in|
|order to gene|rate alt|at|tribute (whi|ch appears before s|rc), we need to use t|he mask|ing capabilit|ies|
|of CM3.|||||||||
||||||<br>||||
|**Cap**|**tioning**||** Masked Pr**|** ompt #1:**<br><img|alt="Photo:<br>A|photo|||
|tak|en of<m|a|sk:0>" sr|c="_{_image_}_">|||||
||||||||||
|**Cap**|**tioning**||** Causal Pr**|** ompt #1:**<br><img|src="_{_image_}_"||||
|tit|le="Pho|t|o:<br>A pho|to taken of|||||
||||<br>|<br>|||||
|We have two|methods|o|f generating|captions given the a|bove prompts. First,|the rel|atively inexp|en-|
|sive method|involves r|u|nning beam|-search with a beam|size of 5 over the pr|oposed|prompts. Fo|r a|
|single image|, we run|bo|th available|prompts and select|the sequence, which|minim|izes the resp|ec-|
|tive CM3 per|plexity. T|he|second me|thod is much more c|ostly and requires sa|mpling|128 captions|for|
|every image|(we note|th|at this is ch|eaper than image gen|eration since image|generat|ion requires|the|
|minimal gen|eration of|2|56 image to|kens while captionin|g is usually on the or|der of|a dozen toke|ns).|
|We then use|CLIP fro|m|Radford et a|l. (2021) to get the t|op ranking caption.|We note|that non-triv|ial|
|captioning b|ehavior w|as|only exhibi|ted in CM3-Large m|odel; therefore, all ev|aluatio|ns will consi|der|
|this singular|model.||||||||
|<br>|<br>||||||||
|We provide|a qualitat|iv|e example|in Figure 6, sourci|ng images and grou|nd truth|captions fr|om|
|MS-COCO|(Lin et al.|,|2014). We|see that CM3 is cap|able of generating n|on-triv|ial semantica|lly|
|coherent cap|tions. Th|at|being said,|most failure cases o|f our proposed zero-|shot ca|ptioning are|due|


7


|an armchair n the shape of an vocado. an|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|armchair||||||
|||||||
|mitating an||||||
|<br>||||||
|avocado.||||||
|||||||
|red car in||||||
|<br>||||||
|the||||||
|||||||
|mountains.||||||
|||||||
|Photo:<br>A||||||
|||||||
|sheep in||||||
|<br>||||||
|owy Artsakh||||||
|<br><br>||||||
|Photo:<br>An||||||
|||||||
|Armenian||||||
|||||||
|urch during||||||
|<br>||||||
|springtime||||||
|||||||
|with clear||||||
|<br>||||||
|skies||||||
|||||||
|e 5: Four samples|for four of th|e prompts usin|g the con|ditional image generation prom|pt with|
|Large. Results we|re selected by|CLIP from a|candidate|set of 32 samples.||
||<br>||<br>|<br>||
||Model||FID|Zero-shot FID||
|||||||
||AttnGAN (Xu e|t al., 2017)|35.49|||
||DM-GAN (Zhu|et al., 2019)|32.64|||
||DF-GAN (Tao e|t al., 2020)|21.42|||
||DM-GAN + CL|(Ye et al., 202|1)<br>20.79|||
||XMC-GAN (Zh|ang et al., 2021|)<br>9.33|||
||LAFITE (Zhou|et al., 2021)|**8.12**|||
||DALL-E (Rame|sh et al., 2021)||_∼_28||
||LAFITE (Zhou|et al., 2021)||26.94||
||GLIDE (Nichol|et al., 2021)||**12.24**||
||<br>|<br>||||
||Unconditional C|M3-Medium||40.65||
||Unconditional C|M3-Medium||36.51||
||Conditional CM|3-Medium||36.78||
||Conditional CM|3-Large||29.56||
|||||||
|2: We compare|FID on MS-C|OCO 256_ ×_|256. Foll|owing Nichol et al. (2021) we|sample|
|ly 30k conditione|d samples for|our models, a|nd compa|re against the entire validation|set. We|
|temperature of 0.|85 for both C|M3 models. W|e use the|implementation available from|Seitzer|
|).||||||
|||||||
|loss of texture f|rom represent|ing images th|rough dis|crete tokens (e.g., the text of th|e train|
|n is blurred, as is|the text on the|bus).||||


8


|Col1|Col2|Col3|the main<br>entrance of the<br>U.S. Department|the white marble<br>exterior|outsi<br>train<br>build|de of a<br>station<br>ing from|
|---|---|---|---|---|---|---|
||||<br>of State in<br>Washington, D.C.|standing atop of<br>its fac¸ade.<br> <br> <br>|<br>acros<br>stree|<br>s the<br>t.|
||||<br>||||
||||a pickup truck<br>parked in a|a large bus<br> <br> <br>|a tal<br>|l red bus<br>|
||||layby on a<br>highway.|parked in a<br>layby<br> <br>|is co<br>some|ing down<br> tracks|
||||||||
||||a man posing for|a man next to a<br>|a man|standing|
||||<br>|<br> <br>|next|to a horse|
||||a photo.|large horse.<br>|on a|beach|
||||||||
||||a U.S. Air Force||||
||||B-52H<br>Stratofortress<br>on the flight|the Austrian<br>Airbus A321<br>aircraft with<br> <br>|a jet<br>flyin|airliner<br>g with a|
||||line at<br>Barksdale Air<br>Force Base,|<br>its Austrian<br>registration<br> <br>|cloud<br>the b|y sky in<br> ackground.|
||||Louisiana||||
|||<br>|||||
|Source Image|Tokenized I|mage<br>|CM3-Caption-Beam|CM3-Caption-CLIP|Gr|ound Truth|
|<br>|<br>|<br><br>|||<br>|<br>|
|Figure 6: We provi|de qualitati|ve sam|ples for zero-shot i|mage-captioning us|ing t|he CM3-Large|
|model. Caption-Bea|m refers to|generat|ing caption using b|eam over prompts, w|hile|Caption-CLIP|
|uses CLIP to get the|top-ranked|caption|from a 128 candida|te set (64 from mask|ed pr|ompt, 64 from|
|causal prompt).|||||||
|<br>|||||||
|Quantitatively we m<br>|easure the<br>|quality<br>|of CM3 zero-shot<br>|captioning by evalu<br>|ating<br>|using BERT-<br>|
|Score2 (Zhang et al.|, 2019) wit|h the R|oBERTa-Large mod|els (Liu et al., 201|9) on|the validation|
|set from MS-COCO|. We opt fo|r the u|se of semantic eval|uation versus classi|cal m|etrics such as|
|BLEU/METEOR be|cause we n|otice th|at the vocabulary a|nd sentence structur|e of|zero-shot cap-|
|tioning with CM3 is|not compa|tible wi|th MS-COCO grou|nd truth labels, alth|ough|the generated|
|content is semantical|ly similar.|We pres|ent our quantitative|result in Table 3. C|M3-La|rge is capable|
|of achieving reasona|ble zero-sh|ot captio|ning performance o|n the MS-COCO da|taset.||
||||<br>|<br><br>|||
||||Precision|Recall<br>F1|||
||||||||
||CM3-Ca|ption-B|eam<br>0.781|0.789<br>0.785|||
||CM3-Ca|ption-C|LIP<br>0.863|0.866<br>0.864|||
||||||||
|Tab|le 3: BERT|Score n|umbers for zero-sho|t captioning with CM|3.||
|<br>|<br>|<br>|<br>|<br>|||
|2We use the open-so|urce BERTS|core at:|https://github.|com/Tiiiger/ber|t_sc|ore. The eval-|
|uation method is: robe|rta-larg|e~~ L~~17~~ n~~|o-idf~~ v~~ersion=0|.3.11(hug~~ t~~rans|=4.1|1.3)~~ f~~ast-t|


9


|4.3 TEXT|MODA|LITY|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|<br>|<br>|||||||||
|CM3 is not on|ly a c|ross-modal m|odel but is fully c|a|pable of ac|tin|g as a|stand-alone langu|age model.|
|This is even re|ﬂect|ed in our data,|where we do not|en|force ever|y d|ocum|ent to have images|; therefore,|
|pure language|mod|eling will also|occur during tra|in|ing. We e|val|uate|our CM3 models on|a wide set|
|of varying lan|guag|e tasks.||||||||
|<br><br>|<br>|<br>||||||||
|4.3.1<br>ENTI|TY D|ISAMBIGUATI|ON|||||||
||<br>|||||||||
|We reproduce|the e|valuation setti|ng described by|D|e Cao et a|l. (|2020)|and Le & Titov (2|018) using|
|the same cand|idate|sets, datasets|and evaluating u|si|ng the InK|B|micro|-F1 metric.||
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>||
|We aim to ﬁn|d a p|rompt capable|of representing t|h|e more ge|ne|ral en|d-to-end entity link|ing task in|
|the CM3 mod|el. F|rom there, a p|roper sequence|sc|oring of t|he|cand|idate set will provi|de us with|
|an approach|to ze|ro-shot entity|disambiguation.||Luckily H|T|ML ba|sed Wikipedia co|ntains very|
|rich annotatio|ns. S|peciﬁcally bel|ow, we show an|ex|ample of|na|turall|y occurring entity l|inking that|
|would occur i|n our|Wikipedia su|bset of CM3 train|in|g data.|||||
|||<br>|<br>|<br>|<br>|||||
|**Ori**|**gina**|**l:**<br>_Manet_|_ho_ writes t|h|at thes|e|kin|gs ruled from||
||||<br>|<br>|<br>|<br>|<br>|||
|<a|titl|e="Memphi|s, Egypt">_M_|_e_|_mphis_</|a>||||
|||<br>|<br>|||||||
|**Pro**|**mpt:**|_Manetho_|writes tha|t|these|ki|ngs|ruled from <|a|
|||<br>|<br>|<br>|<br>|<br>|<br>|||
|tit|le="|<mask:0>"|>_Memphis_</a|>|...<mas|k:|0>|||
|||||||||||
|**Tar**|**get:**|_Manetho_|writes tha|t|these|ki|ngs|ruled from <|a|
|||<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|tit|le="|<mask:0>"|>_Memphis_</a|>|...<mas|k:|0>|Memphis, Egyp|t|
||||<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|Using our sc|oring|approach we|can simply score|t|he** Target**|w|hile s|wapping out the p|ostﬁx after|
|<mask:0>.||||||||||
|||||||||||
||||In-dom|ain|||Ou|t-of-domain||
|**Model Type**||**Method**|**AIDA**||**MSNBC**|**AQ**|**UAINT**|**ACE2004**<br>**CWEB**<br>**W**|**IKI***<br>**Avg.**|
||||<br>|||||||
|||Ganea & Hof|mann (2017)<br>92.2||93.7||88.5|88.5<br>77.9|77.5<br>86.4|
|||Guo & Barbo|sa (2018)<br>89||92||87|88<br>77|84.5<br>86.2|
|||<br><br>Yang et al. (2|018)<br>**95.9**||92.6||89.9|88.5<br>**81.8**|79.2<br>88.0|
|_Direct Supervisi_|_ on_|<br><br><br> <br>Shahbazi et a<br>|<br><br> l. (2019)<br>93.5<br> <br>||92.3<br>||90.1<br>|88.7<br>78.4<br><br>|79.8<br>87.1<br><br>|
|||<br><br><br>Yang et al. (2<br>Le & Titov (|019)<br>93.7<br>  2019)<br>89.6||93.8<br>92.2||88.2<br>90.7|90.1<br>75.6<br>88.1<br>78.2|78.8<br>86.7<br>81.7<br>86.8|
|||Fang et al. (2|019)<br>94.3||92.8||87.5|91.2<br>78.5|82.8<br>87.9|
|||**De Cao et al**|**  . (2020)**<br>93.3||94.3||89.9|90.1<br>77.3|**87.4**<br>88.8|
|||||||||||
|||CM3-Medium|93.5||94.2||90.1|90.4<br>76.5|86.9<br>88.6|
|_Direct Supervisi_|_ on_|_{_<br><br>CM3-Large|94.8||**94.8**||**91.1**|**91.4**<br>78.4|**88.7**<br>**89.8**|
|||||||||||
|||<br>CM3-Medium|78.0||80.1||75.4|81.4<br>68.5|76.2<br>76.6|
|_Self Supervision_|_ (0-Shot_|_ ){_<br><br>CM3-Large|80.1||80.8||77.7|82.8<br>72.4|80.2<br>79.0|
|||||||||||
|||||||||||
|Table 4: Ali|ned|with GENRE’|evaluation, we||se Micro|_ F_1|(In|B) for the named|entity dis-|
|ambiguation|task.|** Bold** indicate|s best model. W|e|note that|alt|<br>    hough|<br>    *WIKI can be th|<br>      ought of as|
|being out-of-|doma|in, given that|English Wikipedi|a|was used|to|pre-tr|ain CM3, it can be|considered|
|in-domain as|well.|||||||||
|<br>|<br>|||||||||
|As an additio<br>|nal d<br>|atapoint for th<br>|e representation<br>|s<br>|learned fr<br>|om<br>|CM3<br>|we completely re<br>|plicate the<br>|
|training and|evalua|tion for the G|ENRE model (|De|Cao et a|l.,|2020)|.3. Speciﬁcally w|e ﬁrst ﬁne-|
|tune CM3 on|the B|LINK data (W|u et al., 2019).|Fo|r the in-d|om|ain s|cenario, we ﬁne-tu|ne CM3 on|
|the AIDA-Co|NLL|dataset (Hoffa|rt et al., 2011).|W|e evaluate|o|n the|AIDA-CoNLL dat|aset for the|
|in-domain sce|nario|and the MSN|BC, AQUAINT,|A|CE2004,|W|NED-|CWEB (CWEB) an|d WNED-|
|WIKI (WIKI|) for|the out-of-do|main scenario (D|e|Cao et al|.,|2020;|Guo & Barbosa,|2018). We|
|present our re|sults|in Figure 4.||||||||
|<br>|<br>|<br>||||||||
|Given the str|ong s|upervision nat|urally available|in|Wikipedi|a|HTM|L, it is unsurprisin|g that CM3|
|shows strong,|non-|trivial zero-sh|ot performance o|n|the name|d e|ntity|disambiguation acr|oss a wide|
|array of name|d ent|ity disambigua|tion tasks.|||||||


[3https://github.com/facebookresearch/GENRE](https://github.com/facebookresearch/GENRE)


10


|Furthermore, the|fine-tuned HTLM-La|rge model outper|form|s previous entity|linking|specific mod-|
|---|---|---|---|---|---|---|
|els to achieve a n|ew SOTA over the be|nchmarked datase|ts.||||
|<br><br>|<br>||||||
|4.3.2<br>ENTITY|LINKING||||||
|<br>|<br>||||||
|We next consider|the more general ent|ity linking task.|We|experiment with t|wo setti|ngs zero-shot|
|assuming we kno|w the location of the|entities and the|full|ﬁne-tuning setting|follow|ing the exact|
|methodology pro|posed in De Cao et al|. (2020). Speciﬁ|cally|for the end-to-en|d Entit|y Linking, we|
|aim to reproduce|the setting of Kolitsa|s et al. (2018). W|e e|valuate using the a|foreme|ntioned_ InKB_|
||||||||
|micro-_F_1 with th|same deﬁned in-do|ain and out-of-d|ma|n datasets as desc|ibed by|De Cao et al.|
|<br>(2020). We use th|<br>  e exact same_ in-doma_|<br>_ in_ and_ out-of-do_|<br>_ main_|<br> datasets as well a|<br>  s evalua|<br>   ting the_ InKB_|
||||||||
|micro-_F_1 on the|GERBIL benchmark|latform (R¨oder|t al|, 2018). Furtherm|ore, we|use the same|
|<br>decoding strategy|<br> for the zero-shot case|<br>   by limiting the g|<br>     ene|<br>     rative tokens to onl|<br>      y availa|<br>       ble candidate|
|entities. Please re|fer to De Cao et al. (2|020) for the full|ﬁne-|tuning setup.|||
|<br>|<br>|<br>|<br>|<br>|||
|For both setting w|e evaluate on seven te|st sets: MSNBC,|Der|czynski (Der) (De|rczynsk|i et al., 2015),|
|KORE 50 (K50)|(Hoffart et al., 2012),|N3-Reuters-128|(R1|28), N3-RSS-500|(R500)|(R¨oder et al.,|
|2014), and OKE|challenge 2015 and 2|016 (OKE15 and|OK|E16) (Nuzzolese e|t al., 20|15).|
|||<br>||<br>|||
|||In-domain||Out-of-domain|||
||**Method**|**AIDA**<br>**MSNB**|**C**<br>**D**|**er**<br>**K50**<br>**R128**<br>**R500**|**OKE15***|**OKE16***<br>**Avg.**|
||||||||
||Hoffart et al. (2011)|72.8<br>65.1|32|.6<br>55.4<br>46.4<br>**42.4**|**63.1**|0.0<br>47.2|
||<br>Steinmetz & Sack (201<br>|3)<br>42.3<br>30.9<br><br>|26<br>|.5<br>46.8<br>18.1<br>20.5<br><br><br><br>|46.2<br>|46.4<br>34.7<br><br>|
|_Direct Supervision_|<br><br><br>Moro et al. (2014)<br>Kolitsas et al. (2018)<br>|48.5<br>39.7<br>82.4<br>72.4<br><br>|2<br>34<br>|.8<br>55.9<br>23.0<br>29.1<br>.1<br>35.2<br>**50.3**<br>38.2<br><br><br><br>|41.9<br>61.9<br>|37.7<br>38.2<br>52.7<br>53.4<br>|
||<br><br><br>Broscheit (2020)<br>Martins et al. (2019)|79.3<br>-<br>81.9<br>-|-|-<br>-<br>-<br><br>-<br>-<br>-|-<br>-|-<br>-|
||van Hulst et al. (2020)_†_|80.5<br>72.4|41|.1<br>50.7<br>49.9<br>35.0|**63.1**|**58.3**<br>56.4|
||De Cao et al. (2020)|**83.7**<br>73.7|**54**|**.1**<br>60.7<br>46.7<br>40.3|56.1|50.0<br>**58.2**|
||<br>||||||
|_Direct Supervision_|_{_<br>CM3-Medium<br>|71.4<br>68.5<br><br>|48<br>|.6<br>58.3<br>44.9<br>41.1<br><br><br><br>|61.9<br>|37.7<br>54.1<br><br>|
||CM3-Large<br>|79.9<br>**74.8**<br><br>|53<br>|.2<br>**62.4**<br>47.1<br>**42.8**<br><br><br><br>|61.9<br>|52.7<br>**59.3**<br><br>|
||<br>CM3-Medium|20.4<br>18.6|20|.1<br>35.1<br>30.6<br>32.1|36.6|0.0<br>24.2|
|_Self Supervision (0-Sho_|_ t){_<br><br>CM3-Large|24.8<br>21.4|2|.6<br>39.0<br>31.1<br>34.9|37.1|0.0<br>26.7|
||||||||
||||||||
|Table 5: We repo<br>|rt Micro_ F_1 on our te<br>|t sets for our ent<br>|ty l<br>|inking task.** Bold** <br>|indicat<br>|s best model.<br>|
|Following De C|<br> o et al. (2020) we us|<br>    a _†_ to indicate|<br> es|<br> lts from the Wiki|edia 2|19 setting as|
||||||||
|opposed to the 20|14 setting (which has|older dump and|few|er entities).|||
|<br>|<br>|<br>|<br>|<br>|||
|We present our re|sults in Table 5. We s|ee that our CM3 ar|e ex|tremely competiti|ve with|entity-linking|
|speciﬁc models a|nd that our CM3-Lar|ge model sets a n|ew|state-of-the-art. F|urtherm|ore, although|
|our zero-shot nu|mbers are substantiall|y worse, they are|still|non-trivial, imply|ing that|CM3 learns a|
|signiﬁcant amoun|t of implicit entity lin|king through our|trai|ning setting.|||
|<br><br>|<br>||||||
|4.3.3<br>SUMMAR|IZATION||||||
||||||||
|We next look at C|M3 performance on th|e zero-shot summ|ari|zation task, speciﬁ|cally w|e replicate the|
|zero-shot evaluat|ion methodology of A|ghajanyan et al.|(202|1). For all summa|rization|benchmarks,|
|we use ROUGE-1|/2/L as our primary m|etrics to stay con|sist|ent with other liter|ature (L|in, 2004). We|
|look at the same|datasets as HTLM.||||||
|<br>|<br>||||||
|**Gigaword** consis|ts of headlines from n|ews articles (Nap|oles|et al., 2012). The|target s|ummaries are|
|relatively short, c|onsisting roughly on|average of 10 BP|E to|kens.|||
|<br>|<br>|<br>|<br>|<br>|||
|**CNN/Dailymail**|(Hermann et al., 201|5) provides multi|-sen|tence target summ|aries c|lose to 3 sen-|
|tences, or roughl|y 50 tokens.||||||
|<br>|<br>||||||
|**Reddit TIFU** (K|im et al., 2018) contai|ns summaries of|Red|dit posts. Speciﬁc|ally, we|use the_ short_|
|subset of data . C|ompared to our other|summarization da|tase|ts, this dataset is h|ighly a|bstractive and|
|not based on new|s articles.||||||
|<br>|<br>||||||
|**XSum** (Narayan|et al., 2018) provides|abstractive single|sen|tence summaries o|f news|articles.|
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|We utilize the sa|me prompts as availa|ble in Aghajanya|n et|al. (2021). We us|e the s|ame available|
|size hints but usi|ng the implicit size-hi|nt methodology d|escr|ibed in § 3.2. Mo|st prom|pts follow the|


11


|same theme of|infilling|either a title|element or|an element describ|ing a headline (eit|her through|
|---|---|---|---|---|---|---|
|attributes or u|sing the|meta tag). For c|ompleten|ess, below is an exa|mple of a prompt|that can do|
|basic summari|zation.||||||
|<br>|||||<br>||
|Model||Gigaword|CNN|/DM<br>Reddit|TIFU<br>XS|um|
|||||<br><br>|<br><br><br>||
|PEGASUS|-0S|23.39/07.59/20.20|32.90/13|.28/29.38<br>14.66/3.|06/10.17<br>19.27/3.0|0/12.72|
|HTLM-Aut|o-NS|27.56/10.17/24.57|33.40/13|.45/30.10<br>06.71/1.|98/07.86<br>15.15/2.5|4/10.91|
|HTLM-Aut|o-S|28.73/11.31/26.49|34.65/14|.54/32.15<br>08.15/2.|92/09.75<br>17.14/3.4|1/13.43|
|HTLM-Ma|nual-S|31.61/10.80/28.60|38.51/16|.10/33.89<br>**15.81/2.**|**98/10.54**<br>22.34/4.1|2/14.56|
||||||||
|CM3-M-Ma|nual|29.15/09.70/27.87|37.16/14|.75/31.42<br>09.56/2.|65/07.48<br>20.14/3.1|5/13.89|
|CM3-L-Ma|nual|**32.12/10.95/28.78**|**38.88/16**|**.27/34.16**<br>12.14/2.|12/07.98<br>**24.86/6.0**|**8/16.32**|
||||||||
|Table 6: CM3|results|on zero-shot sum|marizatio|n. HTLM-Manual|denotes manually|engineered|
|prompts with|size hin|ts, while HTLM-|Auto-S an|d HTLM-Auto-NS|indicate auto-prom|pting with|
|and without si|ze hints|, respectively. Th|e metrics|shown are ROUGE|-1/ROUGE-2/RO|UGE-L, re-|
|spectively. CM|3-Large|sets a new state-|of-the-art o|n three news-based|summarization da|tasets.|
||<br>|<br>|<br>|<br>|<br>|<br>|
|We present ou|r result|s in Table 6. Bo|th CM3 m|odels saw signiﬁca|ntly less text than|the HTLM|
|model, with 2|.7TB of|text. Furthermo|re, the pro|mpts being used w|ere tuned speciﬁc|ally for the|
|HTLM model|and are|being used with n|o changes|for CM3. With the|se challenges, we s|till see that|
|CM3-Large se|ts new s|tate-of-the-art ze|ro-shot su|mmarization for thr|ee datasets. We a|ttribute the|
|performance d|egradati|on in Reddit-TIF|U data to C|M3 pre-training da|ta only containing|CC-NEWS|
|and Wikipedia|, which|will not contain t|he type of|summarizations ne|eded for Reddit-TI|FU.|
|<br><br>|<br>||||||
|5<br>FINE-T|UNING||||||
||||||||
|We next want|to meas|ure the quality of|internal re|presentations for th|e end goal of ﬁne-|tuning. We|
|compare CM3|with a w|ide array of mask|ed langua|ge model derived m|odels such as T5 (R|affel et al.,|
|2019), RoBER|Ta (Liu|et al., 2019), HT|LM (Agha|janyan et al., 2021)|tested on the stan|dard GLUE|
|benchmark (W|ang et|al., 2018). For C|M3 we loo|k at three settings|for ﬁne-tuning; sta|ndard ﬁne-|
|tuning, better|ﬁne-tuni|ng using adversar|ial method|s (Aghajanyan et al|., 2020), and better|ﬁne-tuning|
|over prompts|derived f|rom Aghajanyan|et al. (202|1). We delegate the|speciﬁcs hyper-pa|rameters of|
|the ﬁne-tuning|experim|ents to § A.3|||||
|||<br>|||||
|||MNLI|QQP<br>R|TE<br>QNLI<br>MRPC<br>|CoLA<br>SST-2<br>#|Params|
|||Acc-m/m|m<br>Acc<br>A|cc<br>Acc<br>Acc<br>|Mcc<br>Acc||
||||||||
|T5-Base||87.1/86.|2<br>89.4<br>8|0.1<br>93.7<br>87.5|51.1<br>95.2|220M|
|RoBERTA||90.2/-|92.2<br>8|6.6<br>94.7<br>89.1|68.0<br>96.4|330M|
|RoBERTa|-R3F|91.1/91.|3<br>92.4<br>8|8.5<br>95.3<br>91.6|71.2<br>97.0|330M|
|BART-Lar|ge|89.9/90.|1<br>92.5<br>8|7.0<br>94.9<br>90.4|62.8<br>96.6|400M|
|HTLM||90.3/91.|4<br>92.6<br>8|7.1<br>95.1<br>90.8|64.3<br>96.9|400M|
|HTLM-R|3F|91.4/92.|1<br>92.8<br>8|9.1<br>95.4<br>91.5|69.4<br>97.1|400M|
|HTLM-R|3F-Promp|t<br>91.6/91.|2<br>92.9<br>8|9.4<br>95.7<br>91.7|69.8<br>97.3|400M|
|T5-Large||89.9/89.|6<br>89.9<br>8|7.2<br>94.8<br>89.9|61.2<br>96.3|770M|
|T5-3B||91.4/91.|2<br>89.7<br>9|1.1<br>96.3<br>90.0|67.1<br>97.4|3B|
|T5-11B||92.2/91.|9<br>90.6<br>9|2.8<br>96.9<br>90.4|71.6<br>97.5|11B|
||||||||
|CM3-Med|ium|89.9/89.|7<br>89.6<br>8|9.1<br>93.1<br>86.5|63.1<br>94.9|2.7B|
|CM3-Med|ium-Prom|pt<br>90.8/91.|0<br>89.9<br>9|0.5<br>95.1<br>89.9|66.2<br>96.3|2.7B|
|CM3-Med|ium-RXF|-Prompt<br>90.9/91.|1<br>90.0<br>9|0.7<br>95.3<br>90.0|67.1<br>96.9|2.7B|
||||||||
|CM3-Larg|e|91.1/91.|0<br>89.9<br>9|1.9<br>95.6<br>89.6|64.6<br>94.2|13B|
|CM3-Larg|e-Prompt|91.5/91.|4<br>90.1<br>9|2.4<br>96.2<br>90.1|70.9<br>97.1|13B|
|CM3-Larg|e-RXF-P|rompt<br>91.9/91.|5<br>91.1<br>9|2.5<br>96.4<br>90.3|70.8<br>97.3|13B|
||||||||
|Table 7: Res|ults on t|he GLUE develo|pment set f|or various ﬁne-tuni|ng methods applie|d to CM3.|
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|We present ou|r results|in Table 7. Ov|erall we se|e that both CM3 m|odels are competi|tive against|
|T5 given the s|ame par|ameter setting. Fu|rthermore|, aligning with the|results from Aghaj|anyan et al.|


12


|(2021) we see that placin|g the natural|language ut|terances|of the vari|ous GLUE tasks|into an HTML|
|---|---|---|---|---|---|---|
|prompt while ﬁne-tuning|non-triviall|y improves e|nd-ﬁnet|uning perf|ormance. The fo|llowing exper-|
|iments show that the cau|sally maske|d language|modelin|g approach|is not detrimen|tal to learning|
|ﬁne-tunable representatio|ns, and neit|her is jointly|modelin|g image to|kens.||
|<br><br>|<br>|<br>|||||
|6<br>ETHICAL CONSI|DERATIO|NS|||||
|<br>|||||||
|Prior work has explored|the extent|to which lan|guage m|odels enc|ode harmful gen|der and racial|
|biases that parallel hum|ans through|the Word|Embedd|ing Associ|ation Test (WE|AT) (Caliskan|
|et al., 2017), the Sentenc|e Encoder A|ssociation T|est (SE|AT) (May e|t al., 2019) and|the Grounded-|
|WEAT/Grounded-SEAT|(Ross et al.,|2021) metri|cs for m|ultimodal|language models|(Tan & Celis,|
|2019). Given the genera|tive nature|of CM3 in b|oth the|language a|nd visual moda|lities, we used|
|GWEAT/GSEAT to prob|e our mode|l. Overall,|we eval|uated six b|ias tests for gen|der and seven|
|bias tests for race and fo|und that our|family of C|M3 mod|els show si|gniﬁcantly less|bias than other|
|models, speiciﬁcally Vis|ualBERT (L|i et al., 2019|) and Vi|LBert (Lu|et al., 2019).||
|||<br><br>|<br>|<br><br>|<br><br>||
|||Level<br>Visu|alBert|ViLBert<br>|CM3-Medium<br>C|M3-Large|
||||||||
|C6: M/W, Career/Fam|ily|S|1.05|1.14|0.00|0.98|
|||W|0.54|0.51|0.10|0.12|
||||||||
|C8: Science/Arts, M/|W|S|0.86|1.05|-0.09|0.42|
|||W|0.62|0.14|0.08|0.07|
||||||||
|C11: M/W, Pleasant/|Unpleasant|S|-0.74|-0.84|0.00|-0.64|
|||W|-0.66|-0.31|-0.20|-0.48|
||||||||
|Double Bind: M/W, C|ompetent|S|-0.10|-0.04|0.01|-0.01|
|||W|-0.23|0.30|-0.07|-0.27|
||||||||
|Double Bind: M/W, L|ikeable|S|-0.11|-1.12|-0.24|-0.59|
|||W|-0.60|0.09|0.00|0.10|
||||||||
|Occupations: M/W, O|ccupation|S|0.98|1.82|0.03|0.62|
|||W|0.91|1.80|0.00|0.58|
||||||||
|Total Signiﬁcant Bias|Count|-|5|6|0|2|
|<br>|<br>||||||
|Table 8: Following Ross|et al. (2021)|we present|the resul|ts for all ge|nder bias classe|s on answering|
|the question: “Do joint e|mbeddings c|ontain biases|”? The|numbers in|our table repres|ent effect sizes,|
|and are underlined if the|ir respective|p-values are|below|0.05. Each|bias type and m|odel are tested|
|three times against Word|embeddings|(W) and Se|ntence e|mbeddings|(S).||
|<br>|<br>|<br>|<br>|<br>|<br>||
|We present our empirical|results for g|ender and ra|ce bias in|Table 8 an|d Table 9 respec|tively. Overall,|
|both CM3 have signiﬁca|ntly less bia|s than other|competi|ng models,|most likely due|to our choice|
|to use only Wikipedia an|d CC-NEW|S articles as|training|sources (a|nd recent CC-NE|WS articles at|
|that). Furthermore, we b|elieve the fa|ct that CM3|-Medium|shows no|to very little sig|ns of bias can|
|be an indicator of under-|ﬁtting as the|large model|is, unfor|tunately, a|ble to show som|e bias from our|
|training data.|||||||
|<br>|||||||
|We also qualitatively exp|eriment wit|h whether C|M3 can b|e prompte|d to produce har|mful or objec-|
|tionable images. In gener|al, we notic|ed it was inc|redibly|hard to pro|duce such conte|nt, additionally|
|the lack of the ability to|generate dis|tinctive feat|ures of V|QVAE-G|AN acts to our b|eneﬁt in terms|
|of preserving privacy.|||||||
|<br><br>|||||||
|7<br>RELATED WORK|||||||
|<br>|||||||
|Fundamentally our work|is an extens|ion of the H|TLM w|ork propos|ed by Aghajanya|n et al. (2021)|
|to using the newly propo|sed causally|masked obj|ective, i|ntegrating|images through|VQVAE-GAN|
|tokens, and scaling up o|ver an order|of magnitu|de. Fro|m there, th|e individual cap|abilities of our|
|models are comparable t|o individual|approaches.|||||


13


|Col1|Col2|Level VisualBert ViL|Bert CM3|-Me|diu|m CM3-Large|
|---|---|---|---|---|---|---|
||||||||
|C3: EA/AA, Pleasan|t/Unpleasant|W<br>0.23|0.14||-0.4|4<br>0.10|
|||S<br>0.31<br>-|0.14|-|0.05|7<br>0.05|
||||||||
|C12: EA/AA, Career|/Family|W<br>-0.29<br>|0.43||0.11|7<br>0..23|
|||S<br>-0.54<br>|0.34|-|0.04|9<br>0.28|
||||||||
|C13: EA/AA, Scienc|e/Arts|W<br>0.04|0.21||0.32|5<br>0.12|
|||S<br>0.12|0.68||0.16|9<br>0.465|
||||||||
|Double Bind: EA/A|A, Competent|W<br>0.61|0.87|-|0.53|5<br>0.42|
|||S<br>0.24|0.25||0.|0<br>0.18|
||||||||
|Double Bind: EA/A|A, Likeable|W<br>0.21<br>-|0.23|-|0.53|5<br>0.19|
|||S<br>0.27<br>-|0.74|-|0.53|5<br>0.21|
||||||||
|Occupations: EA/AA|, Occupation|W<br>-0.40<br>|0.02||-0.5|1<br>0.01|
|||S<br>-0.41<br>|0.46||-0.1|7<br>0.38|
||||||||
|Angry Black Woman|Stereotype|W<br>-0.07<br>|0.26||-1.8|9<br>0.21|
|||S<br>-0.50<br>|0.47||0.|0<br>-0.10|
||||||||
|Total Signiﬁcant Bia|s Count|-<br>4|5|||1<br>3|
|<br>|<br>||||||
|Table 9: Following Ros|s et al. (2021)|we present the results fo|r all racial|bias|cl|asses on answering|
|the question: “Do joint|embeddings c|ontain biases”? Our table|uses the sa|me|ann|otations as Table 8.|
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|For example, the condit|ional and unc|onditional image generati|on capabili|ties|of|our model are most|
|similar in approach to|DALL-E, whic|h trains a left-to-right ca|usal model|ove|r th|e concatenation of|
|textual tokens and VQ-|VAE visual to|kens (Ramesh et al., 202|1). At the s|ame|tim|e, the use of auto-|
|regressive modeling in|entity linking|and disambiguation was|proposed b|y t|he|GENRE in De Cao|
|et al. (2020).|||||||
|<br>|||||||
|The method of tokeniz|ing non-discre|te modalities to use stan|dard seque|nce|mo|deling approaches|
|have been extensively e|xplored with|DALL-E for images, Juke|box for Mu|sic|(Dh|ariwal et al., 2020)|
|and vq-wav2vec for Sp|eech (Baevski|et al., 2019).|||||
|<br><br>|||||||
|8<br>CONCLUSION|||||||
||||||||
|In this paper, we presen|t the CM3 mod|el, a causally masked trai|ned langua|ge m|od|el that is capable of|
|non-trivial zero-shot pe|rformance on|a wide range of zero-shot|uni- and c|ross|-m|odal tasks. We ﬁrst|
|describe a new sequenc|e modeling ob|jective we call causally|masked, ena|bli|ng b|oth full generative|
|modeling with bidirecti|onal context.||||||
|<br>|<br>||||||
|Through extensive exp|erimentation,|we show that as a single|model CM|3 c|an|be prompted to re-|
|cover the functionality|of many other|models being able to do|image gen|erat|ion,|image captioning,|
|unconditional image ge|neration, and|more. Empirically we im|prove over|sta|te-|of-the-art zero-shot|
|summarization, entity l|inking, entity|disambiguation, highligh|ting the str|uct|ure|from the hypertext|
|during training. We s|how that repre|sentations learned by C|M3 are not|onl|y u|seful for zero-shot|
|prompting but for ﬁne-|tuning by ﬁne|-tuning CM3 and state-o|f-the-art fo|r en|tity|linking and entity|
|disambiguation in gene|ral, all while st|aying highly competitive|with T5 m|odel|s o|n the GLUE bench-|
|mark.|||||||
||||||||
|REFERENCES|||||||
||||||||
|Armen Aghajanyan, Ak|shat Shrivasta|va, Anchit Gupta, Naman|Goyal, Lu|ke Z|ett|lemoyer, and Sonal|
|Gupta. Better ﬁne-tu|ning by reduci|ng representational collap|se._ arXiv p_|_ repr_|_ int_|_  arXiv:2008.03156_,|
|2020.|||||||
||||||||
|Armen Aghajanyan, D|mytro Okhonk|o, Mike Lewis, Mandar J|oshi, Hu X|u, G|arg|i Ghosh, and Luke|
|Zettlemoyer. Htlm:|Hyper-text pr|e-training and prompting|of languag|e m|od|els._ arXiv preprint_|


_arXiv:2107.06955_, 2021.


14


|ikel Artetxe, Shru|ti Bhosale,|Naman Goyal, Todo|r Mihaylov, Myl|e Ott, Sam Shl|eifer, Xi Victo-|
|---|---|---|---|---|---|
|ria Lin, Jingfei D|u, Srinivasa|n Iyer, Ramakanth|Pasunuru, et al.|Efﬁcient large|scale language|
|modeling with mi|xtures of ex|perts._ arXiv preprin_|_ t arXiv:2112.106_|_  84_, 2021.||
|<br>|<br>|<br>||<br>||
|exei Baevski, Ste|ffen Schnei|der, and Michael A|uli. vq-wav2ve|c: Self-supervis|ed learning of|
|discrete speech re|presentatio|ns._ arXiv preprint ar_|_  Xiv:1910.05453_,|2019.||
|<br>|<br>|<br>|<br>|<br>||
|andeep Baines, S|hruti Bhosa|le, Vittorio Caggian|o, Naman Goya|l, Siddharth Go|yal, Myle Ott,|
|Benjamin Lefaud|eux, Vitaliy|Liptchinsky, Mike|Rabbat, Sam Sh|eiffer, Anjali Sr|idhar, and Min|
|Xu. Fairscale: A|general pur|pose modular pytor|ch library for hig|h performance|and large scale|
|training. https:|//githu|b.com/facebook|research/fa|irscale, 202|1.|
|||||<br>|<br>|
|eba Birhane, Vin|ay Uday Pr|abhu, and Emmanue|l Kahembwe. M|ultimodal datas|ets: misogyny,|
|pornography, and|malignant s|tereotypes._ arXiv pr_|_ eprint arXiv:211_|_  0.01963_, 2021.||
|<br>|<br>|<br>||<br>||
|muel Broscheit. I|nvestigating|entity knowledge in|bert with simpl|e neural end-to-|end entity link-|
|ing._ arXiv preprin_|_ t arXiv:200_|_  3.05473_, 2020.||||
|||<br>||||
|m B Brown, Ben|jamin Man|n, Nick Ryder, Mela|nie Subbiah, Jar|ed Kaplan, Praf|ulla Dhariwal,|
|Arvind Neelakant|an, Pranav|Shyam, Girish Sastr|y, Amanda Aske|ll, et al. Langu|age models are|
|few-shot learners.|_ arXiv prep_|_ rint arXiv:2005.141_|_  65_, 2020.|||
|<br>|||<br>|||
|lin Caliskan, Joa|nna J. Brys|on, and Arvind Nar|ayanan. Semanti|cs derived auto|matically from|
|language corpora|contain hu|man-like biases._ Scie_|_ nce_, 356:183 – 1|86, 2017.||
|<br>|<br>|<br>|<br>|<br>||
|cola De Cao, Gaut|ier Izacard,|Sebastian Riedel, an|d Fabio Petroni.|Autoregressive|entity retrieval.|
|_arXiv preprint ar_|_ Xiv:2010.00_|_ 904_, 2020.||||
|||<br>||||
|on Derczynski, D|iana Mayna|rd, Giuseppe Rizzo,|Marieke Van Er|p, Genevieve G|orrell, Rapha¨el|
|Troncy, Johann P|etrak, and K|alina Bontcheva. A|nalysis of named|entity recogniti|on and linking|
|for tweets._ Inform_|_ ation Proc_|_ essing & Manageme_|_  nt_, 51(2):32–49,|2015.||
|<br>|||<br>|<br>||
|cob Devlin, Ming|-Wei Chang|, Kenton Lee, and|Kristina Toutano|va. Bert: Pre-tr|aining of deep|
|bidirectional trans|formers for|language understan|ding._ arXiv prep_|_ rint arXiv:1810._|_  04805_, 2018.|
|<br>|<br>|<br>|<br>||<br>|
|afulla Dhariwal, H|eewoo Jun,|Christine Payne, Jon|g Wook Kim, Al|ec Radford, and|Ilya Sutskever.|
|Jukebox: A gener|ative model|for music._ arXiv pr_|_ eprint arXiv:200_|_  5.00341_, 2020.||
|<br>|<br>|<br>||<br>||
|trick Esser, Robin|Rombach,|and Bjorn Ommer.|Taming transfor|mers for high-re|solution image|
|synthesis. In_ Proc_|_ eedings of_|_  the IEEE/CVF Conf_|_   erence on Comp_|_    uter Vision and_|_     Pattern Recog-_|
|_nition_, pp. 12873–|12883, 202|1.||||
|<br>|<br>|<br>||||
|exander R Fabbri|, Simeng H|an, Haoyuan Li, H|aoran Li, Marja|n Ghazvinineja|d, Shaﬁq Joty,|
|Dragomir Radev,|and Yashar|Mehdad. Improvin|g zero and few-s|hot abstractive|summarization|
|with intermediate|ﬁne-tuning|and data augmentat|ion._ arXiv prepri_|_ nt arXiv:2010.1_|_  2836_, 2020.|
|<br>|<br>|<br>|<br>||<br>|
|eng Fang, Yanan|Cao, Qian|Li, Dongjie Zhang,|Zhenyu Zhang,|and Yanbing Li|u. Joint entity|
|linking with deep|reinforcem|ent learning. In_ T_|_ he World Wide W_|_  eb Conference_|, pp. 438–447,|
|2019.||||||
|||||||
|tavian-Eugen Ga|nea and Tho|mas Hofmann. Dee|p joint entity dis|ambiguation wi|th local neural|
|attention._ arXiv p_|_ reprint arXi_|_  v:1704.04920_, 2017|.|||
|||<br>|<br>|||
|aochen Guo and|Denilson B|arbosa. Robust na|med entity disam|biguation with|random walks.|
|_Semantic Web_, 9(|4):459–479,|2018.||||
|<br>|<br>|<br>||||
|rl Moritz Herma|nn, Tomas|Kocisky, Edward Gr|efenstette, Lasse|Espeholt, Will|Kay, Mustafa|
|Suleyman, and P|hil Blunso|m.<br>Teaching machi|nes to read and|comprehend.<br>I|n_ Advances in_|
|_neural informatio_|_ n processin_|_ g systems_, pp. 1693–|1701, 2015.|||
|||<br>|<br>|||
|artin Heusel, Hub|ert Ramsa|uer, Thomas Untert|hiner, Bernhard|Nessler, and Se|pp Hochreiter.|
|Gans trained by a|two time-s|cale update rule con|verge to a local|nash equilibrium|._ Advances in_|
|_neural informatio_|_ n processin_|_ g systems_, 30, 2017.||||


15


|hannes Hoffart, Moham|ed Amir Y|osef, Ilaria Bordin|o, Hagen Fu¨rstena|u, Manfre|d Pinkal, Marc|
|---|---|---|---|---|---|
|Spaniol, Bilyana Tanev|a, Stefan T|hater, and Gerhard|Weikum. Robust|disambigu|ation of named|
|entities in text. In_ Proc_|_ eedings of_|_  the 2011 Confere_|_   nce on Empirical_|_    Methods i_|_     n Natural Lan-_|
|_guage Processing_, pp. 7|82–792, 2|011.||||
|<br>|<br>|<br>||||
|hannes Hoffart, Stephan|Seufert, D|at Ba Nguyen, Ma|rtin Theobald, and|Gerhard|Weikum. Kore:|
|keyphrase overlap relat|edness for|entity disambiguati|on. In_ Proceedin_|_ gs of the 2_|_  1st ACM inter-_|
|_national conference on_|_  Informatio_|_  n and knowledge m_|_   anagement_, pp. 5|45–554, 20|12.|
||||<br>|<br>|<br>|
|ed Kaplan, Sam McCa|ndlish, To|m Henighan, Tom|B Brown, Benja|min Chess,|Rewon Child,|
|Scott Gray, Alec Radfo|rd, Jeffrey|Wu, and Dario A|modei. Scaling|laws for n|eural language|
|models._ arXiv preprint_|_  arXiv:2001_|_  .08361_, 2020.||||
|||<br>||||
|eongchang Kim, Hyun|woo Kim,|and Gunhee Kim.|Abstractive sum|marization|of reddit posts|
|with multi-level memor|y networks|._ arXiv preprint ar_|_  Xiv:1811.00783_, 2|018.||
|<br>|<br>|<br>|<br>|<br>||
|ederik P Kingma and Ji|mmy Ba.|Adam: A method|for stochastic opt|imization.|_ arXiv preprint_|
|_arXiv:1412.6980_, 2014.||||||
|<br>|<br>|||||
|kolaos Kolitsas, Octavia|n-Eugen G|anea, and Thomas|Hofmann. End-to|-end neural|entity linking.|
|_arXiv preprint arXiv:18_|_ 08.07699_,|2018.||||
||<br>|<br>||||
|ong Le and Ivan Titov.|Improving|entity linking by m|odeling latent rel|ations bet|ween mentions.|
|_arXiv preprint arXiv:18_|_ 04.10637_,|2018.||||
||<br>|<br>||||
|ong Le and Ivan Titov.|Boosting e|ntity linking perfor|mance by leverag|ing unlabe|led documents.|
|_arXiv preprint arXiv:19_|_ 06.01250_,|2019.||||
||<br>|<br>||||
|ike Lewis, Yinhan Liu,|Naman G|oyal, Marjan Ghaz|vininejad, Abdel|rahman M|ohamed, Omer|
|Levy, Ves Stoyanov,|and Luke|Zettlemoyer.<br>Ba|rt:<br>Denoising se|quence-to-|sequence pre-|
|training for natural la|nguage ge|neration, translati|on, and compreh|ension.<br>|_arXiv preprint_|
|_arXiv:1910.13461_, 201|9.|||||
|<br>|<br>|||||
|unian Harold Li, Mark Y|atskar, Da|Yin, Cho-Jui Hsieh|, and Kai-Wei Ch|ang. Visua|lbert: A simple|
|and performant baseline|for vision|and language._ arX_|_ iv preprint arXiv:_|_  1908.0355_|_  7_, 2019.|
|<br>|<br>|<br>|||<br>|
|in-Yew Lin. Rouge: A|package f|or automatic evalu|ation of summarie|s. In_ Text_|_ summarization_|
|_branches out_, pp. 74–81|, 2004.|||||
|<br>|<br>|||||
|ung-Yi Lin, Michael M|aire, Serge|Belongie, James|Hays, Pietro Pero|na, Deva R|amanan, Piotr|
|Doll´ar, and C Lawrenc|e Zitnick.|Microsoft coco:|Common objects|in context.|In_ European_|
|_conference on computer_|_  vision_, pp|. 740–755. Springe|r, 2014.|||
||<br>|<br>|<br>|||
|nhan Liu, Myle Ott, N|aman Goya|l, Jingfei Du, Man|dar Joshi, Danqi|Chen, Om|er Levy, Mike|
|Lewis, Luke Zettlemoy|er, and Ves|elin Stoyanov. Rob|erta: A robustly o|ptimized b|ert pretraining|
|approach._ arXiv preprin_|_ t arXiv:19_|_  07.11692_, 2019.||||
|||<br>||||
|sen Lu, Dhruv Batra, D|evi Parikh|, and Stefan Lee.|Vilbert: Pretraini|ng task-ag|nostic visiolin-|
|guistic representations f|or vision-a|nd-language tasks.|_ arXiv preprint ar_|_  Xiv:1908.0_|_  2265_, 2019.|
|<br>|<br>|<br>||||
|dro Henrique Martins,|Zita Mari|nho, and Andr´e FT|Martins.<br>Joint|learning o|f named entity|
|recognition and entity li|nking._ arX_|_ iv preprint arXiv:1_|_  907.08243_, 2019.|||
|<br>|<br>||<br>|||
|andler May, Alex Wang|, Shikha B|ordia, Samuel R. B|owman, and Rac|hel Ruding|er. On measur-|
|ing social biases in sent|ence encod|ers._ ArXiv_, abs/190|3.10561, 2019.|||
|<br>|<br>|<br>|<br>|||
|drea Moro, Alessandro|Raganato,|and Roberto Navi|gli. Entity linking|meets wor|d sense disam-|
|biguation: a uniﬁed app|roach._ Tra_|_ nsactions of the As_|_  sociation for Com_|_   putational_|_    Linguistics_, 2:|
|231–244, 2014.||||||
|<br>|||<br>||<br>|
|urtney Napoles, Matth|ew R Gor|mley, and Benjam|in Van Durme.<br>|Annotated|gigaword.<br>In|
|_Proceedings of the Join_|_  t Worksho_|_  p on Automatic Kn_|_    owledge Base Co_|_     nstruction_|_     and Web-scale_|
|_Knowledge Extraction (_|_ AKBC-WE_|_ KEX)_, pp. 95–100,|2012.|||


16


|ashi Narayan, Sha|y B Coh|en, and|Mirella La|pata.|Don’t give me the details, just the sum-|
|---|---|---|---|---|---|
|mary! topic-aware|convolu|tional n|eural netw|orks for|extreme summarization._ arXiv preprint_|
|_arXiv:1808.08745_,|2018.|||||
|<br>|<br>|||||
|ex Nichol, Prafulla|Dhariw|al, Adity|a Ramesh,|Pranav|Shyam, Pamela Mishkin, Bob McGrew,|
|Ilya Sutskever, and|Mark Ch|en. Glid|e: Towards|photor|ealistic image generation and editing with|
|text-guided diffusio|n model|s._ arXiv_|_ preprint ar_|_  Xiv:211_|_  2.10741_, 2021.|
|<br>|<br>|<br>|||<br>|
|drea Giovanni Nu|zzolese,|Anna|Lisa Gent|ile, Va|lentina Presutti, Aldo Gangemi, Dar´ıo|
|Garigliotti, and Rob|erto Na|vigli. Op|en knowle|dge extr|action challenge. In_ Semantic Web Evalu-_|
|_ation Challenges_, p|p. 3–15.|Springer|, 2015.|||
|<br>|<br>|<br>|<br>|||
|yle Ott, Sergey Edu|nov, Al|exei Bae|vski, Ange|la Fan,|Sam Gross, Nathan Ng, David Grangier,|
|and Michael Auli.|fairseq:|A fast,|extensible|toolki|t for sequence modeling._ arXiv preprint_|
|_arXiv:1904.01038_,|2019.|||||
|<br>|<br>|||||
|am Paszke, Sam G|ross, Fra|ncisco M|assa, Adam|Lerer,|James Bradbury, Gregory Chanan, Trevor|
|Killeen, Zeming L|in, Nata|lia Gime|lshein, Lu|ca Anti|ga, et al.<br>Pytorch: An imperative style,|
|high-performance d|eep lear|ning libr|ary._ Advan_|_ ces in n_|_  eural information processing systems_, 32:|
|8026–8037, 2019.||||||
|<br>||||||
|even T Piantadosi.|Zipf’s w|ord freq|uency law|in natu|ral language: A critical review and future|
|directions._ Psychon_|_ omic bu_|_ lletin &_|_  review_, 21(|5):1112|–1130, 2014.|
||||<br>|<br>|<br>|
|ec Radford, Jeffrey|Wu, Re|won Chil|d, David L|uan, Da|rio Amodei, and Ilya Sutskever. Language|
|models are unsuper|vised mu|ltitask l|earners._ Op_|_ enAI B_|_ log_, 1(8):9, 2019.|
|<br>|<br>|<br>|<br>||<br>|
|ec Radford, Jong|Wook Ki|m, Chris|Hallacy, A|ditya|Ramesh, Gabriel Goh, Sandhini Agarwal,|
|Girish Sastry, Ama|nda Ask|ell, Pam|ela Mishki|n, Jack|Clark, et al. Learning transferable visual|
|models from natura|l langua|ge super|vision._ arX_|_ iv prepr_|_ int arXiv:2103.00020_, 2021.|
|<br>|<br>|<br>|<br>||<br>|
|lin Raffel, Noam S|hazeer, A|dam Ro|berts, Kath|erine Le|e, Sharan Narang, Michael Matena, Yanqi|
|Zhou, Wei Li, and P|eter J L|iu. Explo|ring the li|mits of t|ransfer learning with a uniﬁed text-to-text|
|transformer._ arXiv_|_ preprint_|_  arXiv:19_|_  10.10683_,|2019.||
||||<br>|<br>||
|itya Ramesh, Mikh|ail Pavlo|v, Gabri|el Goh, Sco|tt Gray|, Chelsea Voss, Alec Radford, Mark Chen,|
|and Ilya Sutskever.|Zero-sh|ot text-to|-image gen|eration|._ arXiv preprint arXiv:2102.12092_, 2021.|
|<br>|<br>|<br>|<br>|<br>|<br>|
|ichael R¨oder, Ricar|do Usbe|ck, Seb|astian Hell|mann,|Daniel Gerber, and Andreas Both. N3-a|
|collection of datas|ets for n|amed en|tity recogn|ition a|nd disambiguation in the nlp interchange|
|format. In_ LREC_, p|p. 3529–|3533, 20|14.|||
|<br>|<br>|<br>|<br>|||
|ichael R¨oder, Ricar|do Usbe|ck, and|Axel-Cyrill|e Ngon|ga Ngomo. Gerbil–benchmarking named|
|entity recognition a|nd linkin|g consis|tently._ Sem_|_ antic W_|_ eb_, 9(5):605–625, 2018.|
|<br>|<br>|<br>|<br>||<br>|
|ndace Ross, Boris|Katz, a|nd Andr|ei Barbu.|Measur|ing social biases in grounded vision and|
|language embeddin|gs._ ArXi_|_ v_, abs/20|02.08911,|2021.||
|<br>|<br>|<br>|<br>|<br>||
|aximilian Seitzer. p|ytorch-ﬁ|d: FID S|core for Py|Torch.|https://github.com/mseitzer/|
|pytorch-fid, A|ugust 20|20. Vers|ion 0.2.1.|||
|<br>|<br>|<br>|<br>||<br>|
|med Shahbazi, Xi|aoli Z F|ern, Rez|a Ghaeini,|Rasha|Obeidat, and Prasad Tadepalli.<br>Entity-|
|aware elmo: Learni|ng conte|xtual en|tity represe|ntation|for entity disambiguation._ arXiv preprint_|
|_arXiv:1908.05762_,|2019.|||||
|<br>|<br>|||||
|dine Steinmetz and|Harald|Sack. Se|mantic mul|timedia|information retrieval based on contextual|
|descriptions. In_ Ext_|_ ended S_|_ emantic_|_  Web Confer_|_  ence_, p|p. 382–396. Springer, 2013.|
|<br>||||<br>|<br>|
|Chern Tan and L.|Elisa Cel|is. Asse|ssing socia|l and in|tersectional biases in contextualized word|
|representations. In|_ NeurIPS_|, 2019.||||
|<br>||<br>||||
|ing Tao, Hao Tang,|Songso|ng Wu,|Nicu Sebe,|Xiao-Y|uan Jing, Fei Wu, and Bingkun Bao. Df-|
|gan: Deep fusion g|enerative|adversa|rial networ|ks for te|xt-to-image synthesis._ arXiv:2008.05865_,|
|2020.||||||


17


|hannes M van Hulst|, Faegheh Has|ibi, Koen Der|cksen, Krisztian Balog, and Arjen P de Vries. Rel:|
|---|---|---|---|
|An entity linker sta|nding on the sh|oulders of gia|nts. In_ Proceedings of the 43rd International ACM_|
|_SIGIR Conference_|_ on Research a_|_  nd Developme_|_   nt in Information Retrieval_, pp. 2197–2200, 2020.|
||||<br>|
|ex Wang, Amanpre|et Singh, Julia|n Michael, Fel|ix Hill, Omer Levy, and Samuel Bowman. GLUE:|
|A multi-task bench|mark and ana|lysis platform|for natural language understanding. In_ Proceed-_|
|_ings of the 2018 E_|_  MNLP Worksh_|_   op BlackboxN_|_   LP: Analyzing and Interpreting Neural Networks_|
|_for NLP_, pp. 353–3|55, Brussels,|Belgium, Nov|ember 2018. Association for Computational Lin-|
|guistics. doi: 10.1|8653/v1/W18|-5446. URL|https://www.aclweb.org/anthology/|
|W18-5446.||||
|||||
|dell Wu, Fabio Petr|oni, Martin Jo|sifoski, Sebas|tian Riedel, and Luke Zettlemoyer. Scalable zero-|
|shot entity linking|with dense ent|ity retrieval._ a_|_ rXiv preprint arXiv:1911.03814_, 2019.|
|<br>|<br>|<br>|<br>|
|o Xu, Pengchuan Z|hang, Qiuyua|n Huang, Han|Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong|
|He. Attngan: Fine|-grained text t|o image gene|ration with attentional generative adversarial net-|
|works._ arXiv:1711._|_ 10485_, 2017.|||
||<br>|||
|yuan Yang, Xiaotao|Gu, Sheng Li|n, Siliang Tan|g, Yueting Zhuang, Fei Wu, Zhigang Chen, Guop-|
|ing Hu, and Xiang|Ren. Learning|dynamic con|text augmentation for global entity linking._ arXiv_|
|_preprint arXiv:190_|_ 9.02117_, 2019|.||
||<br>|<br>||
|Yang, Ozan Irsoy,|and Kazi She|faet Rahman.|Collective entity disambiguation with structured|
|gradient tree boosti|ng._ arXiv pre_|_ print arXiv:18_|_  02.10229_, 2018.|
|<br>|<br>||<br>|
|i Ye, Xiulong Yan|g, Martin Taka|c, Rajshekhar|Sunderraman, and Shihao Ji. Improving text-to-|
|image synthesis usi|ng contrastive|learning._ arX_|_ iv:2107.02423_, 2021.|
|<br>|<br>|<br>|<br>|
|n Zhang, Jing Yu K|oh, Jason Bal|dridge, Hongl|ak Lee, and Yinfei Yang. Cross-modal contrastive|
|learning for text-to-|image genera|tion._ arXiv:21_|_ 01.04702_, 2021.|
|<br>|<br>|<br>|<br>|
|anyi Zhang, Varsha|Kishore, Felix|Wu, Kilian Q|Weinberger, and Yoav Artzi. Bertscore: Evaluat-|
|ing text generation|with bert._ arX_|_ iv preprint ar_|_  Xiv:1904.09675_, 2019.|
|<br>|<br>||<br>|
|fan Zhou, Ruiyi Zh|ang, Changyo|u Chen, Chun|yuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu,|
|Jinhui Xu, and Ton|g Sun. Laﬁte|: Towards lan|guage-free training for text-to-image generation.|
|_arXiv:2111.13792_,|2021.|||
|<br>|<br>|||
|infeng Zhu, Pingbo|Pan, Wei Che|n, and Yi Yan|g. Dm-gan: Dynamic memory generative adver-|
|sarial networks for|text-to-image|synthesis._ arX_|_ iv:1904.01310_, 2019.|


18


|A A|PPEND|IX|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
|A.1<br>M|ODEL|ARCHITECTURE|||||||||
||<br>|<br>|||||||||
|For mo|del archi|tecture we use the sam|e exact a|rchitect|ure for CM3|-|Mediu|m and CM|3-Large|as the|
|dense 2|.7B and|13B models described|in Artetx|e et al.|(2021).||||||
|||||<br>|<br>||||||
||||||CM3-Large||CM3-|Medium|||
||||||||||||
|||–decoder-embed-dim|||5120|||2560|||
|||–decoder-output-dim|||5120|||2560|||
|||–decoder-input-dim|||5120|||2560|||
|||–decoder-ffn-embed-d|im||20480|||10240|||
|||–decoder-layers|||40|||32|||
|||–decoder-normalize-b|efore||True|||True|||
|||–decoder-attention-he|ads||40|||32|||
|||–share-decoder-input-|output-e|mbed|True|||True|||
|||–decoder-learned-pos|||False|||False|||
||||||||||||
|||Table 10: FairSeq|architect|ure desi|gnation for|CM|3 mo|dels|||
|||<br>|<br>|<br>|||||||
|A.2<br>U|NIFOR|MITY OF VQVAE-GA|N TOKE|NS|||||||
|||<br>|<br>||||||||
|We plot|a histo|gram of all image tok|ens in a s|ubset o|f our data s|pa|nning|100k tok|ens. We|see a|
|somewh|at clear|uniformity in tokens u|sed.||||||||
||<br>|<br>|<br>||||||||
||Figu|re 7: Histogram of VQ|-VAE-G|AN Tok|ens in the C|M3|Train|ing Datas|et.||
||<br>|<br>|<br>|<br>|||||||
|A.3<br>F|INETUN|ING GLUE HYPER-P|ARAMET|ERS|||||||
|||<br>|||||||||
|For our|ﬁne-tun|ing GLUE related exp|eriments|with th|e RXF meth|o|d we u|se the fol|lowing|hyper-|
|paramet|ers.||||||||||
|||<br>|||||||||
||Hyper P|arameter<br>MNLI|QNLI|QQP|SST-2|R|TE|MRPC|CoLA||
|<br>|<br>|<br><br> <br>|||||||||
||Learnin|g Rate<br>5e-6|5e-6|5e-6|5e-6|1e|-5|1e-5|1e-5||
||Max Up|dates<br>123873<br>|33112|113272|20935<br>|31|20<br>|2296<br>|5336||
||Max Se|ntences<br>8|8|32|32|8||16|16||
|||<br><br>|||||||||
|||Table 11: Task speci|ﬁc hyper|paramet|ers for GLU|E|exper|iments|||


19


|Hyper parameter|Value|Col3|Col4|
|---|---|---|---|
|<br>||||
|Optimizer|Adam<br>|<br>||
||<br>|Hyper parameter<br>|Value|
|Adam-betas|(0.9, 0.98)<br>|<br>||
|Adam-eps|1e-6<br>|_λ_|[0.1,|
|LR Scheduler|polynomial decay<br>|Noise Types|[_U_,_ N_|
|Dropout<br>|0.1<br>|_σ_|1_e −_|
|Weight Decay|0.01|||
|Warmup Updates|0.06 * max updates|||



Table 12: Hyper parameters for fine-tuning experiments on GLUE


20


