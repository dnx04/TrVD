# Enhancing vulnerability detection via AST decomposition and neural sub-tree encoding

## Abstract

The explosive growth of software vulnerabilities poses a serious threat to the system security and has become one of the urgent problems of the day. However, existing vulnerability detection methods are still faced with limitations in reaching the balance between detection accuracy, efficiency and applicability. Following a divide-and-conquer strategy, this paper proposes TrVD (abstract syntax Tree decomposition based Vulnerability Detector) to disclose the indicative semantics implied in the source code fragments for accurate and efficient vulnerability detection. To facilitate the capture of subtle semantic features, TrVD converts the AST of a code fragment into an ordered set of sub-trees of restricted sizes and depths with a novel decomposition algorithm. The semantics of each sub-tree can thus be effectively collected with a carefully designed tree-structured neural network. Finally, a Transformer-style encoder is utilized to aggregate the long-range contextual semantics of all sub-trees into a vulnerability-specific vector to represent the target code fragment. The extensive experiments conducted on five large datasets consisting of diverse real-world and synthetic vulnerable samples demonstrate the performance superiority of TrVD against SOTA approaches in detecting the presence of vulnerabilities and pinpointing the vulnerability types. The ablation studies also confirm the effectiveness of TrVD's core designs.

## 1. Introduction

The main contributions of this paper are as follows:

- We propose a novel vulnerability detection method called TrVD, with elaborate design in every formulation phase, to make it an accurate, efficient and more practical approach that applies to code fragments. Specifically, to advance its ability in detecting well-hidden vulnerabilities and the specific vulnerability types, it combines novel AST decomposition, attention-augmented subtree encoding, and context-aware semantic aggregation to enable the distillation of subtle yet vulnerability-specific semantics from the target code fragments, which, to the best of our knowledge, is the first of this kind in the DL-based vulnerability detection works.
- We conduct extensive experiments to evaluate the performance of TrVD, the effectiveness of its different composing modules, and the runtime overheads. The empirical studies show that TrVD outperforms state-of-the-art DL-based methods on the majority datasets in terms of accuracy, F1 and precision in either detecting the existence of vulnerability or identifying the specific vulnerability types. The superiority of TrVD's core designs, such as AST decomposition and sub-tree encoding, are also confirmed through ablation studies.
- We contribute a new dataset to facilitate vulnerability detection researches. This dataset contains 264,822 C/C++ functions, each of which is labeled with either specific CWE ID or non-vulnerable ground truth. The dataset and the source code for TrVD implementation have been made available to facilitate future benchmarking or comparison.

The rest of this paper is organized as follows. Section 2 discusses the background knowledge and the core of our approach. The related works are reviewed in Section 3. Section 4 details the designs of TrVD. Section 5 presents the experimental evaluations and analyses. Section 6 discusses possible threats to validity issues, the limitations and interesting future works we plan to extend. Finally, Section 7 concludes this work.

## 2. Preliminaries and the core

### 2.1. Basic terms and concepts

**Vulnerability** refers to the presence of weaknesses, defects, or security bugs within computer systems that can be potentially exploited by attackers. These vulnerabilities enable unauthorized access, such as stealing sensitive data, or allow attackers to execute arbitrary actions, such as installing malware, on the targeted computer system. Such vulnerabilities can manifest across diverse aspects, including software code, hardware components, configurations, or design. This study focuses on detecting potential vulnerabilities embedded in the code.

**CWE**, which stands for Common Weakness Enumeration, is a community-developed vulnerability list (CWE, 2023). It offers a standardized and structured way to identify and categorize these vulnerabilities, assigning a unique identifier to each one. For example, CWE-119 refers to the infamous "Buffer Overflow". Following different levels of conceptual abstraction, CWE organizes the vulnerabilities in a tree-like hierarchical structure, where low-level CWE-IDs are affiliated to higher-level CWE-IDs. For instance, CWE-787 that denotes "Out-of-bounds Write" and CWE-125 that denotes "Out-of-bounds Read", are both lower-level types affiliated with CWE-119.

We further provide brief descriptions of some commonly used code representations in program analysis tasks as follows.

**Code tokens** refer to lexical tokens in a program that carry specific semantics. They encompass identifiers (e.g., variable and function names), keywords, separators (e.g., punctuation and delimiters), and operators (e.g., arithmetic and logic operators). A code fragment can be straightforwardly represented as a sequence or set of tokens through lexical parsing.

**AST**, or Abstract Syntax Tree, is a fundamental code representation that organizes code elements into a tree structure. Tree leaves correspond to primary code elements, such as variable types, symbols, and operators, while non-leaf nodes represent a restricted set of code structures, such as expressions and loops (Tang, Shen et al., 2022). Compared to lexically parsed code tokens, AST also naturally manifests the syntactic structure of source code in addition to its lexical information.

**CFG**, or Control Flow Graph, is a directed graph where each node represents a basic block of statements, and each edge represents the flow of control between the blocks within a function. The CFG is constructed by identifying the control dependencies in the AST. It generally requires the code to be functionally complete and compilable to precisely generate the CFG.

**PDG**, or Program Dependency Graph (Harrold, Malloy, & Rothermel, 1993), is another graph representation of the code that emphasizes both data and control dependencies between code elements. Similar to the CFG, it can be constructed on the basis of AST. During construction, certain code details are abstracted to reveal control and data dependencies more explicitly.

**CPG**, or Code Property Graph (Yamaguchi, Golde, Arp, & Rieck, 2014), presents a more synthesized hybrid graph representation of code, which integrates information derived from the AST, CFG and PDG.

### 2.2. Problem to solve

Various methods have been developed to detect vulnerabilities (Cui et al., 2022), including static, dynamic, concolic analysis, and fuzzing approaches. As we will delve into further in the related work section, TrVD falls into the learning-based paradigm within the static detection series. In this paradigm, the vulnerability detection task is formulated as a classification problem.

Specifically, in the training phase, a classifier M is learned from a set of training samples with ground-truth labels through code representation construction, feature extraction, and model training. In the detection phase, when presented with a potentially unseen piece of source code C , the same procedures of code representation construction and feature extraction are performed. The trained classifier M then predicts the presence of vulnerability, or further pinpoint the specific vulnerability type.

We aim to develop TrVD into an accurate, efficient and more practically applicable vulnerability detection method on source code. As the first training step, the selection of an appropriate initial code representation is essential to achieve the goals TrVD aims for. In this respect, TrVD resorts to operating on the AST, a choice that is believed to benefit TrVD from the following three major aspects.

- **Availability.** Other well-adopted code representations, as discussed in the previous section, include CFG (Cao, Sun, Bo, Wei, & Li, 2021; Cheng et al., 2019), PDG (Li, Wang, & Nguyen, 2021), and diverse graph-based variants (Chakraborty et al., 2021; Ding et al., 2022). These representations more explicitly delineate control or data dependencies among code elements, which, however, are challenging to be derived precisely when facing non-compilable or incomplete code fragments. Therefore, they may not always be feasible for vulnerability detection. By contract, AST can be constructed easily for any code fragment, e.g., a file, a function, or a single statement. Such wide availability ensures TrVD's practical applicability.
- **Efficiency.** Compared with code representations (e.g., CFG, PDG and code gadgets) (Li, Zou, Xu, Jin et al., 2022; Li et al., 2018; Tang, Hu et al., 2022; Zou, Wang, Xu, Li, & Jin, 2021) that require relatively complex and time-consuming control or dependency analysis, constructing an AST from code is much more straightforward and light-weight, thus contributing to the efficiency of the whole detection method.
- **Semantic Comprehensiveness.** Those artificially-created code representations (e.g., PDG and XFG (Cheng, Wang, Hua, Xu, & Sui, 2021)), tend to emphasize specific aspects of the code, such as the control flow or data dependency relationships (Harrold et al., 1993). However, they often lose some important information during the transformation process, which incurs semantic losses, particularly when representing incomplete code fragments. Differently, AST makes the highly structured nature of source code, where the underlying syntax regarding statements and expressions is directly available (Hou, Chen, & Ye, 2022); hence, the original code fragment can be exactly restored from it. In other words, AST provides more comprehensive, richer, and more precise code semantics and enables TrVD not to miss any suspicious vulnerability implications and enhance the detection accuracy.

However, as one may have noticed, despite the impressive properties of AST in facilitating DL-based vulnerability detection, its semantics are not so obvious as those purposefully-crafted representations such as CFG and PDG that intentionally make semantics more explicit. Thus, the major challenge faced by TrVD is how to comprehensively and effectively learn the suspicious semantics that are indicative of vulnerabilities from the expressive yet implicating AST.

### 2.3. The core

To fully exploit the potential of the AST and effectively address the challenge of its usage as previously discussed, which significantly impacts on vulnerability detection capability, TrVD follows the classic divide-and-conquer strategy.

**Divide.** Encoding the entire AST using graph/tree-based neural networks, such as GCN (Kipf & Welling, 2016), GAT (Veličković et al., 2017), Tree-LSTM (Tai, Socher, & Manning, 2015), and TBCNN (Mou, Li, Zhang, Wang, & Jin, 2016), may greatly degrade in capturing the long-range dependencies with over-smoothed embedding when dealing with large/deep tree structures (Zhang et al., 2019). Due to resource restrictions, it is also highly infeasible for these time- and memory-consuming models to handle such large/deep ASTs directly. In this regard, TrVD opts to divide the whole AST into an ordered set of sub-trees via a decomposition algorithm, each of which corresponds to a fine-grained intact code unit in the original code fragment. This yields two advantages here: (1) each sub-tree contains a limited (much smaller) number of nodes and controllable tree depth, and can be manipulated by general tree or graph neural networks in a cost-inexpensive way; and (2) more importantly, this offers an opportunity to better grasp the subtle semantics indicating vulnerability information by focusing on a local code unit of restricted size, which otherwise may be easily overlooked.

**Conquer.** TrVD adopts a "comprehensive-acquisition followed by critical-point-focus" scheme to achieve the effective extraction of indicative and suspicious vulnerability semantics. To be specific, TrVD first processes each sub-tree with a newly designed tree-structured neural network to map the semantics of its corresponding code unit into a numerical vector. Based on the learned sub-tree embeddings, TrVD then leverages a Transformer-based backbone to attend differentially to more and less important sub-trees to discover vulnerability patterns with self-attention mechanism, and fuse the long-range contextual semantics of all sub-trees into a dense and vulnerability-specific numerical vector to represent the target code fragment. In this sense, the way of TrVD in code representation learning is similar to slice-based methods (Li, Zou, Xu, Chen et al., 2022; Li, Zou, Xu, Jin et al., 2022; Li et al., 2018; Zou et al., 2021), where code gadgets generated by slicing the relevant statements according to points of interest (e.g., library/API functions) accommodate information useful for learning local features and helping pinpoint vulnerability patterns through regular NLP models (e.g., LSTM). But sub-trees refined in TrVD are more structured and less ambiguous, while the Transformer devised for attention-augmented sub-tree aggregation enables better contextualized embedding to improve vulnerability detection performance.

## 3. Related work

We focus on reviewing the closely-related static source code vulnerability detection works that fall into the following categories, including code similarity matching based methods, static rule based methods and learning based methods.

### 3.1. Similarity matching-based methods

This line of methods (Cui, Hao, Jiao, Fei, & Yun, 2021; Jang, Agrawal, & Brumley, 2012; Kang, Son, & Heo, 2022; Kim, Woo, Lee, & Oh, 2017b; Pham, Nguyen, Nguyen, & Nguyen, 2010) generally calculate the similarity between the abstracted representations of the target code and the code of known vulnerability, to determine the absence/presence of potential vulnerabilities within the target code. The specific code representations adopted by existing approaches for similarity matching are varying, such as tokens (Jang et al., 2012; Kim et al., 2017b), trees (Pham et al., 2010; Yamaguchi, Lottmann, & Rieck, 2012), graphs (Cui et al., 2021), their hybrids (Li et al., 2016), well-crafted signatures (Kang et al., 2022; Xiao et al., 2020) and embeddings (Sun et al., 2021) learned by siamese neural networks. Overall, they excel at detecting recurring vulnerabilities introduced by code or library reusing, rather than general or undisclosed vulnerabilities.

### 3.2. Static rule-based methods

The static rule-based methods proceed by scanning the target source code with a mass of elaborately defined vulnerability rules or patterns. The typical static analyzers, such as FlawFinder (A. Wheeler, 2014), Cppcheck (Marjamaki, 2013), Infer (Infer, 2013), CodeChecker (CodeChecker, 2013) and Checkmarx (Checkmarx, 2022) fall into this category. However, the rules used to scan the code are mainly summarized by experienced security experts, with each of them describing a specific vulnerability scenario. Thus, as confirmed by existing studies (Goseva-Popstojanova & Perhinschi, 2015; Lipp et al., 2022), this line of methods are only effective for vulnerabilities of certain types (e.g., buffer-overflows, use-after-frees) and tend to report a large portion of false positives or false negatives in detecting vulnerabilities due to the diversity of the real-world code and possible vulnerability patterns. Recent researches (Cheng et al., 2021; Li, Zou, Xu, Jin et al., 2022; Wu et al., 2022) that follow the learning-based roadmap show superior vulnerability detection performance over the rule-based methods.

### 3.3. Learning-based methods

The early works (Ghaffarian & Shahriari, 2017; Perl et al., 2015; Younis, Malaiya, Anderson, & Ray, 2016) generally adopt traditional machine learning algorithms to train detection models with the representative features summarized by experts, such as code complexity metrics, code churns, imports and calls, and developer activities (Bosu, Carver, Hafiz, Hilley, & Janni, 2014). As a whole, these engineered features are inadequate in indicating the presence of vulnerabilities. Also, most existing methods are limited to the in-project rather than the general-purpose vulnerability detection.

With great successes of deep learning in a wide range of applications, the recent works that achieve SOTA vulnerability detection performances all employ deep learning schema. Resting on the initial code representations that the neural models operate on, existing DL-based methods can be generally summarized into three categories: token-based methods, slice-based methods and graph-based methods. The token-based methods convert the code into one or a set of token sequences by lexically parsing the code (Russell et al., 2018) or linearizing its abstract syntax tree (Lin, Zhang, Luo, Pan, & Xiang, 2017; Lin et al., 2018) in certain manners (e.g., pre-order traversal, leaf-node path traversal with sampling) so as to preserve additional contextual and structural information. They are efficient due to the light-weight code processing and the low-overhead code encoding with sequence-oriented neural network structures, but generally incapable of accurately detecting well-hidden vulnerabilities. The slice-based methods (Li, Zou, Xu, Chen et al., 2022; Li, Zou, Xu, Jin et al., 2022; Li et al., 2018; Zou et al., 2021) represent the code as code gadgets (Li et al., 2018; Tang, Hu et al., 2022; Zou et al., 2021) (i.e., groups of statements that have data or control dependencies) or the derivatives (e.g., SeVCs (Li, Zou, Xu, Jin et al., 2022), iSeVCs (Li, Zou, Xu, Chen et al., 2022)). They are inefficient for enforcing the time-consuming data and control dependency analysis as well as performing slicing; furthermore, as the slices are generated against certain code elements (i.e., program points of interest, such as the library/API function calls, the array usages, etc.), these methods become inapplicable in detecting vulnerabilities that are irrelevant to these elements. The graph-based methods (Cheng et al., 2021; Li, Wang et al., 2021) encode with graph neural networks (Shi et al., 2022; Wu et al., 2021) the widely-used conventional graph-structured code representations, such as CFG (Cao et al., 2021; Cheng et al., 2019), PDG (Li, Wang et al., 2021), their hybrids (Zhou et al., 2019), or elaborately derived structures, such as CPG (Chakraborty et al., 2021; Ding et al., 2022) and XFG (Cheng et al., 2021). They are more sophisticated and time-consuming in either constructing these graph structures or training well-behaving models. Compared with these methods, our TrVD aims to reach and has indeed achieved a good balance between detection accuracy, efficiency and applicability.

To provide a clearer insight into the distinctiveness of TrVD in contrast to the aforementioned DL-based approaches, Table 1 presents a summary of their fundamental characteristics. This includes their major categories, the initial code representations they work with, the processing employed to transform the initial representation, the final format for model input after transformation, the applied learning model, and a qualitative assessment of the computational overhead.

## 4. Design of TrVD

Fig. 1 depicts the overall architecture of TrVD, which involves the training phase that learns a classifier M on the vulnerability datasets with well-labeled ground truths (which will be discussed in Section 5.1.1), and the detection phase that uses the trained classifier M to output a prediction. During both phases, the target source code fragment is fed through five main modules in turn to get processed: (1) AST Generator, which normalizes the code fragment in a semantic-preserving fashion and constructs its AST with a parser; (2) AST Decomposer, which converts the AST into an ordered set of subtrees with a decomposition algorithm; (3) Comprehensive Semantic Collector, which iteratively distills the conveyed semantics from each sub-tree into a real-valued vector with a tree-structured neural encoder; (4) Suspicious Semantic Focuser, which aggregates all the collected subtree semantics into a contextualized embedding vector that attends to more vulnerability-specific semantics with Transformer-based model; (5) Vulnerability Detector, which implements a multi-layer perceptron (MLP) with softmax layer trained with cross-entropy loss to predict an output. The details of these five modules are discussed as follows.

### 4.1. AST generator

Instead of directly constructing the AST for the original source code fragment, we opt to perform light-weight syntactic abstractions that normalize its code tokens with the following considerations. First, different coding habits naturally present among different programmers, where their personalized naming styles for function and variable names barely indicate any vulnerability information, but may act as noises to deteriorate the code representation learning by immersing TrVD in too many unnecessary details. Second, to facilitate the distillation of semantics from the sub-trees with neural encoders, each tree node that corresponds to a certain token needs to be pre-mapped into embedding space; however, retaining all the original tokens may raise two issues: (1) enlarging the vocabulary size that accordingly increases the embedding complexity and decreases the embedding quality; (2) intensifying the out-of-vocabulary (OOV) impact and degrading subsequent vulnerability-indicative semantics extraction.

Table 1
Major characteristics of the DL-based vulnerability detection approaches.

| Category | Approaches | Initial representation | Processing | Model inputs | Model | Overhead |
|-------------|---------------------------------------------------------------------------------------|------------------------|---------------|----------------------------|-----------------------|----------|
| Token-based | Russell et al. ( 2018 ) | Code | Tokenization | Token sequence | CNN | Low |
| | Lin et al. ( 2017 ) | AST | Serialization | Token sequence | BILSTM | Low |
| | Lin et al. ( 2018 ) | AST | Serialization | Token sequence | BILSTM | Low |
| Slice-based | Li et al. ( 2018 ) | DDG | Slicing | Code Gadgets | BILSTM | High |
| | Tang, Hu et al. ( 2022 ) | PDG | Slicing | Code Gadgets | CNN+Attention | High |
| | Zou et al. ( 2021 ) | PDG | Slicing | Code Attention and Gadgets | BILSTM | High |
| | Li, Zou, Xu, Jin et al. ( 2022 ) | PDG | Slicing | SeVCs | BiGRU | High |
| | Li, Zou, Xu, Chen et al. ( 2022 ) | PDG | Slicing | iSeVCs | BiRNN+Attention | High |
| Graph-based | Cheng et al. ( 2019 ) | CFG | – | CFG | GCN | High |
| | Li, Wang et al. ( 2021 ) | PDG | – | PDG | FA-GCN | High |
| | Zhou et al. ( 2019 ) | CPG | – | CPG | GGNN | High |
| | Chakraborty et al. ( 2021 ) | CPG | – | CPG | GGNN | High |
| | Cheng et al. ( 2021 ) | PDG | Slicing | XFG | GNN | High |
| | Ding et al. ( 2022 ) | CPG | – | CPG | Transformer + GGNN | High |
| Tree-based | TrVD | AST | Decomposition | Sub-Tree Sequence | AttRvNN + Transformer | Medium |

More specifically, after removing the comment lines from the original code, we normalize its remaining code tokens, by reassigning each user-defined variable and function with a new symbolic name in the form of VAR_i and FUN_i respectively, where i is the order of the variable or function for its first appearance in the code fragment, while the remaining tokens, including reserved words, literals, library function names, and punctuations remain intact.

Fig. 2 illustrates a normalization processing example that is performed on a short code snippet. It should be noted that the normalization performed according to the above defined abstraction rules is semantic-preserving; that is, no semantic losses are enforced during the normalization phase, which is important to ensure the reliability of the whole method. On the basis of the normalized code fragment, we then leverage tree-sitter, a widely-adopted C/C++ parsing library, to obtain its abstract syntax tree.

### 4.2. AST decomposer

As discussed in Section 2.3 about the problems raised by encoding the AST as a whole for vulnerability detection, this module decomposes the AST into finer-grained code units that are much more affordable to subsequent semantic encoders, and make the subtle semantics conveyed in the code more explicit to the neural encoders. Formally, given the AST T , the task of our decomposer is to divide it into an ordered set D with a policy zeta , where D is composed of code units of restricted sizes.

Different policies have been designed to decompose the AST to facilitate source code analysis tasks, which can be summarized into three major categories according to their outputs.

- Single token sequence. Traversal algorithms are generally utilized to flatten the AST into a token sequence, including pre-order traversal (Tang, Shen et al., 2022; Zhang, Wang, Zhang, Sun, & Liu, 2020) and in-order traversal (Svyatkovskiy, Zhao, Fu, & Sun-daresan, 2019). To avoid information loss, SBT (Hu, Li, Xia, Lo, & Jin, 2018) and SPT (Niu et al., 2022) further improve these methods by appending additional symbols to indicate the parent-child relationships to ensure that the linearized sequence can be unambiguously mapped back to the AST, which, however, complicates the sequence with larger size. Another method code2seq (Alon, Levy, & Yahav, 2019) concatenates the sampled paths collected between the leaf nodes to form a single token sequence, which is often too long to be efficiently handled by the neural encoders for semantic extraction, especially for vulnerability detection task.
- Token sequence set. PathTrans (Kim, Zhao, Tian, & Chandra, 2021) maps the AST to a set of root-paths, each of which is comprised of the tokens obtained by traversing either up from a leaf node to the root, or down from the root to a leaf node (Jiang, Zheng, Lyu, Li, & Lyu, 2021), and points out the longitudinal context that a specific code element (i.e., the leaf node) resides in. The root-paths can serve as inputs for learning token embeddings; however, as each path only contains fragmentary code elements, the incomplete semantics learned from them may degrade the training and the detection performance.
- Sub-trees. Instead of linearizing the AST as token sequences, methods such as ASTNN (Zhang et al., 2019) and Infercode (Bui, Yu, & Jiang, 2021) decompose it into a set of sub-trees to leverage the syntactic structures. Different granularities are adopted to split the AST, where ASTNN produces non-overlapping sub-trees that correspond to code statements, while Infercode produces sub-trees with overlaps that correspond to smaller code elements such as an expression.

```

1: void bad()
2: {
3: wchar_t * data;
4: wchar_t dataBadBuffer[10];
5: wchar_t dataGoodBuffer[10+1];
6: if(GLOBAL_CONST_TRUE)
7: {
8: /* FLAW: Set a pointer to a buffer that does not leave room for a NULL
9: terminator when performing string copies in the sinks */
10: data = dataBadBuffer;
11: data[0] = L'\0'; /* null terminate */
12: }
13: {
14: wchar_t source[10+1] = SRC_STRING;
15: size_t i, sourceLen;
16: sourceLen = wcslen(source);
17: /* Copy length + 1 to include NUL terminator from source */
18: /* POTENTIAL FLAW: data may not have enough space to hold source */
19: for (i = 0; i < sourceLen + 1; i++)
20: {
21: data[i] = source[i];
22: }
23: printWLine(data);
24: }
25: }

```

```

1: void FUN0()
2: {
3: wchar_t * VAR0;
4: wchar_t VAR1[10];
5: wchar_t VAR2[10+1];
6: if(GLOBAL_CONST_TRUE)
7: {
10: VAR0 = VAR1;
11: VAR0[0] = L'\0';
12: }
13: {
14: wchar_t VAR3[10+1] = SRC_STRING;
15: size_t VAR4, VAR5;
16: VAR5 = wcslen(VAR3);
19: for (VAR4 = 0; VAR4 < VAR5 + 1; VAR4++)
20: {
21: VAR0[VAR4] = VAR3[VAR4];
22: }
23: FUN1(VAR0);
24: }
25: }

```

Our decomposer follows a similar policy as ASTNN to divide the AST into a sequence of non-overlapping sub-trees at the granularity of statement, but dismisses the finer-grained expression or single-token sub-tree used in Infercode. The reasons behind this are that (1) the latter method generates a very long sub-tree sequence where the long-range context weakens the intrinsic relationships among sub-trees, and (2) each sub-tree can only accommodate incomplete or even useless code information. Also, ASTNN always splits a compound statement (e.g., *if* statement) into a header sub-tree and a set of sub-trees that corresponds to the statements within its body, which disrupts the local semantic and structural integrity. Different from ASTNN, our decomposer only decomposes these compound statements when necessary. Specifically, we set up a checker to inspect the size and depth of the sub-tree rooted at each compound statement node, and further split it only when either its size or depth exceeds certain threshold. As such, a good balance can be reached by producing sub-trees with light weight yet preserving relatively complete local semantics.

Algorithm 1 illustrates the detailed processing steps of our decomposer. Given the root node, it proceeds by recursively splitting out sub-trees through a visitor and a constructor. The visitor performs pre-order traversal along the AST to pass each node to the constructor for node type inspection, compound statement decomposition, and sub-tree construction, where the constructed sub-trees are appended to the set in order. Here, the order for each sub-tree in the final set is decided by the position of its corresponding code in the original code fragment. An illustrative example is given in Fig. 3 to facilitate the understanding of our AST decomposition algorithm as well as its differences from ASTNN. More specifically, taking the normalized code in Fig. 3(a) as input, our algorithm decomposes its AST into 12 sub-trees ordered by their appearances in the code. Compared with ASTNN, the differences of our algorithm are also illustrated: for the *if* compound statement from line 6 to line 10 in the code, our algorithm generates one integrated sub-tree, while ASTNN splits it into three trivial sub-trees for each sub-statement as framed by the blue dotted lines in Fig. 3(a).

### 4.3. Comprehensive semantic collector

Given the ordered sub-tree set D = \{\tau_1, \tau_2, \dots, \tau_n\} , where n denotes the number of sub-trees produced with the decomposer and tau_i denotes the i th sub-tree, the collector iteratively processes each of them with a tree-structured recursive neural network (RvNN) to collect its latent semantics. This follows a bottom-up style, where RvNN recursively

#### Algorithm 1. Decomposing AST into Statement Sub-trees
###### Input: T : an abstract syntax tree. alpha : the maximum depth allowed for a sub-tree. beta : the maximum number of nodes allowed in a sub-tree.

###### Output: D : an ordered sub-tree set.

function TreeSplitting( \tau, D, alpha, beta, C ) node \leftarrow \text{getRoot}(\tau) ▷ get the root node of current tree

if node is a statement **then**

if node \in C **then**

if \text{trSize}(node) < alpha and \text{trDepth}(node) < beta **then** ▷

construct a sub-tree rooted at node and append it to D $t \leftarrow \text{subTreeConstructor}(node)$ D \leftarrow D \oplus t else t \leftarrow \text{subTreeConstructor}(node.header) $D \leftarrow D \oplus t$

for each child node child of node.body **do**

TreeSplitting( child, D )

end for

end if

else t \leftarrow \text{subTreeConstructor}(node) $D \leftarrow D \oplus t$

end if

else

for each child node child of node **do**

TreeSplitting( child, D )

end for

end if

end function D \leftarrow \emptyset $C \leftarrow [\text{'FuncDef', 'If', 'For', 'While', 'Dowhile', 'Switch', 'Try', 'Catch'}]$

▷ the type list of the compound statements

TreeSplitting( T, D, alpha, beta, C )

return D ---

aggregates the local node representations from the leaves all the way up to the root to derive a global representation for the entire sub-tree. Formally, let S_j denote the direct children of node v_j in the sub-tree where \sigma is the sigmoid function; \odot denotes the element-wise product, the W s, U s and b s are the weights and biases to be jointly learned with other parameters during the training process; e_j is the embedding vector that corresponds to the token in node v_j , which is learned with skip-gram (Mikolov, Chen, Corrado, & Dean, 2013) algorithm on a corpus of paths collected from the ASTs to enable a higher initialization for training than straightforward yet computationally intractable one-hot embedding schema; h_s denotes the children's aggregated semantics by pooling the hidden states of all the nodes in S_j .

There are multiple ways to pool the children's hidden states into a single vector h_s , such as direct sum, average or max pooling. However, these pooling operations generally overlook the differences among the child nodes in the aspect of their semantics. In order to capture such semantic differences, we introduce an attention mechanism to strengthen hidden state aggregation over h_s with more expressive power, which allows the pooling process to pay more attention on more informative child nodes. Specifically, we adapt the context/query attention (Yang et al., 2016) to calculate an importance weight for each child node s \in S_j , on the basis of which a weighted sum of the children's hidden states is computed as their aggregated semantics h_s . This attention-augmented calculation can be formalized as: $h_s = \sum_{s \in S_j} \alpha_s \cdot h_s, \quad (5)$ $$\alpha_s = \frac{\exp(\beta_s^T \mu_w)}{\sum_{k \in S_j} \exp(\beta_k^T \mu_w)}, \quad (6)$$ $\beta_s = \tanh (W_s h_s + b_s) \quad (7)$ where \alpha_s is the normalized attention weights, which indicates the relative importance of each node and essentially controls how much information can flow from each child node to its parent node; h_s is the hidden state of node s \in S_j ; W_s and b_s are the respective weight matrix and bias vector corresponding to the hidden state h_s , both of which can be jointly learned with the context/query vector \mu_w during the training process.

After iteratively calculating the hidden state of every node in the sub-tree using the attention-augmented tree-structured RvNN in a bottom-up way, we take the hidden state of its root node as the sub-tree's final semantic encoding vector. In this regard, for the sub-tree tau_i , its semantic encoding can be denoted as h_{\tau_i} = h_{v_0} = \mathcal{H}(\tau_i) , where h_{v_0} is the hidden state of the root node in sub-tree tau_i , and \mathcal{H}(\cdot) is the semantic collector with composite functions presented in the Eqs. (4)~(7).

### 4.4. Suspicious semantics focuser

Based on the semantic encodings H_D = \{h_{\tau_1}, h_{\tau_2}, \dots, h_{\tau_n}\} systematically gathered from the ordered sub-tree set D by the collector, TrVD further utilizes the Focuser to aggregate them to obtain the semantic encoding h_T for the entire AST T , with the help of the Transformer architecture. It is well known that Transformer is currently the most prominent neural network architecture that achieves state-of-the-art performance against other sequence-oriented neural networks, such as the RNN, LSTM and TextCNN, in many NLP and program analysis tasks. By taking advantage of its multi-head self-attention mechanism, TrVD is able to capture the long-range contextual semantic relationships among the sub-trees, and pay more attention on the vulnerability-specific semantics.

The Focuser only uses the encoder part of the Transformer architecture to produce the AST's encoding h_T . More specifically, the sub-tree

representation vectors H_D are first fed into a series of vertically stacked Transformer blocks; each block comprises a multi-head attention layer, a normalization layer and a position-wise feed-forward layer, which are used for contextual semantic relationship capturing and vulnerability-indicative semantic focusing. To implement the multi-head attention layer, the core operations are the self-attention score calculations, which can be computed within each head with the following equations: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}, \quad (8)$ $$\mathbf{Q} = \mathbf{W}^Q H_D, \mathbf{K} = \mathbf{W}^K H_D, \mathbf{V} = \mathbf{W}^V H_D \quad (9)$$ where $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \in \mathbb{R}^{m \times d_v}$ is the attention score matrix, in which each row g_i is a d_v -dimensional attention vector for the corresponding row in the input matrix H_D ; \mathbf{Q} \in \mathbb{R}^{m \times d_k} , \mathbf{K} \in \mathbb{R}^{m \times d_k} and \mathbf{V} \in \mathbb{R}^{m \times d_v} are three matrices derived by multiplying the input H_D with trainable weight matrices; and \sqrt{d_k} is a scaling factor to ensure more stable gradients by avoiding producing too large softmax results. After that, the distilled semantic vectors (i.e., the outputs of the stacked Transformer blocks) corresponding to the input sequence H_D at different time steps, are manipulated with the max-pooling operation to obtain the dense vector h_T to represent the target code fragment.

### 4.5. Vulnerability detector

By taking the vector representation h_T of the entire AST as input, the Detector module predicts the vulnerable/non-vulnerable result or the specific vulnerability type, with a multi-layer perceptron. The predicted probability distribution \text{MLP}_\theta(h_T) is computed using a softmax function to facilitate minimizing the cross-entropy loss between the prediction and ground truth, where \theta are the parameters from all modules that are optimized during the model training.

## 5. Experiments and evaluations

To evaluate TrVD, we seek to answer the following *Research Questions* (RQs):

- **RQ1: Performance on Vulnerability Presence Detection.** What is the performance of TrVD in detecting the presence of vulnerabilities against other methods?
- **RQ2: Performance on Vulnerability Type Detection.** What is the performance of TrVD in pinpointing the type of vulnerabilities compared with other methods?
- **RQ3: Ablation Study on TrVD's Core Components.** How do our designs including AST decomposition, sub-tree encoding and AST encoding contribute to the performance of TrVD?
- **RQ4: Efficiency Analysis of TrVD.** How about the runtime overhead of TrVD against other vulnerability detection methods?

### 5.1. Experimental setup

#### 5.1.1. Datasets

To evaluate our method, we construct a dataset consisting of C/C++ functions on the basis of the Software Assurance Reference Dataset (SARD), a vulnerability database that is widely used as the source for producing experimental samples. The programs in SARD are collected as a mixture of the production, academic and synthetic code, with each program associated with a "bad", "good" or "mixed" label. Each "bad" program generally contains one vulnerable function, while each "good" program only contains non-vulnerable functions that are fixed or patched from their vulnerable versions. Each "mixed" program organizes both the vulnerable function and its patched versions within one program.

Table 2
Some statistical information of the datasets. The usage scenario column depicts whether TrVD will be evaluated on them for vulnerable presence detection or further vulnerability type detection.

| Dataset | #Vulnerable | #Non-Vulnerable | Dataset type | Usage scenario | |
|----------------|-------------|-----------------|--------------|----------------|------|
| | | | | Presence | Type |
| Ours | 98,181 | 166,641 | Synthetic | ✓ | ✓ |
| D2A | 2795 | 2444 | Real-World | ✓ | ✗ |
| VulDeePecker | 9739 | 150,409 | Hybrid | ✓ | ✗ |
| Reveal | 2240 | 20,494 | Real-World | ✓ | ✗ |
| muVulDeePecker | 43,119 | 138,522 | Hybrid | ✗ | ✓ |

Based on the raw C/C++ programs in SARD, we first process them with the tree-sitter parsing library to identify and collect the vulnerable and their patched functions, and remove the ones that fail in converting into ASTs with our AST Generator. Then, each collected function is associated with its vulnerable/non-vulnerable ground truth, and each vulnerable function is further associated with a CWE-ID (Common Weakness Enumeration Identifiers) (CVE, 2023) to indicate the specific vulnerability type it contains, based on the name of the file that each function resides in. As a result, we obtain a dataset consisting of 264,822 well-labeled functions, where 166,641 are non-vulnerable and 98,181 are vulnerable across 118 CWE-IDs.

The programs in SARD are basically synthetic. Thus, we also evaluate TrVD on four other vulnerability datasets that are used in the comparison works, including:

- **D2A** (Zheng et al., 2021), a real-world vulnerability dataset curated from several open-source software projects, such as FFmpeg, httpd, Libav, LibTIFF, Nginx and OpenSSL, by the IBM Research team. A differential analysis technique is designed to label the samples.
- **VulDeePecker** (Li et al., 2018), a hybrid dataset that consists of real-world samples from the National Vulnerability Database (NVD) and synthetic samples from the SARD.
- **Reveal** (Chakraborty et al., 2021), a real-world dataset curated by tracking the past vulnerabilities from the Linux Debian Kernel and Chromium.
- **muVulDeePecker** (Zou et al., 2021), a remarkably similar dataset as VulDeePecker which is also curated from the NVD and the SARD, but with the vulnerable samples distributed to 40 different vulnerability types.

In our experiments, the first three datasets which correlate each of their samples with a vulnerable or non-vulnerable ground truth, are used in the vulnerability presence detection task; while the muVulDeePecker dataset which contains well-labeled vulnerability type information is used in the vulnerability type detection task. Our constructed dataset is used for both tasks. A statistical summary on these datasets regarding quantities of their vulnerable and non-vulnerable samples as well as their usage scenarios in our evaluations, is reported in Table 2.

#### 5.1.2. Implementation and experimental settings

We implement TrVD fully in python. Its AST Generator uses the widely-adopted tree-sitter for extracting ASTs from the code fragments written in the C/C++ programming language. The default parameters alpha and beta that guide the AST Decomposer to split out sub-trees are set to 8 for tree depth and 40 for tree size, respectively. The skip-gram algorithm for learning the nodes' initial token embeddings adopts the implementation provided in gensim, with the embedding dimension set to 128. The rest neural network modules are implemented with PyTorch, where the dimension of the GRU hidden state in the Collector is set to 128, and the hidden dimension of the Transformer block in the Focuser is set to 128 as well.

For the experimental settings, we split the datasets into training, validation and testing sets with the proportion of 8:1:1. Since the ratios between the vulnerable and non-vulnerable samples in some of the datasets are fairly imbalanced, we undersample the non-vulnerable samples to the quantity of vulnerable samples within each dataset, to maintain a balanced training set. As for the testing and validation sets, we use the real ratios presented in the original datasets. We train our models with an initial learning rate 1e-3 (which is decreased by 0.8 every 10 epochs), a batch size of 32, and the Adam optimizer. In each epoch, the training samples are shuffled and accuracy on the validation set is calculated. The early stopping mechanism is leveraged to stop the training when the validation accuracy no longer rises for 5 epochs. Finally, the model that shows the best accuracy is taken as the final detection model, with which the performance evaluation results are reported on the testing set. All the experiments are conducted on a Linux server, with two 2.1 GHz Intel Xeon Silver-4310 CPUs, 128 GB RAM and two NVIDIA RTX3090 GPUs.

#### 5.1.3. Evaluation metrics

Following most of existing learning-based vulnerability detection methods, we also use Accuracy, Precision, Recall and F1 -score as the metrics, for fair performance comparison against other methods. Note that, in the vulnerability type detection experiments, accuracy refers to total accuracy, while precision, recall and F1 -score refer to weighted-average precision, recall and F1 -score respectively.

### 5.2. Experimental results

#### 5.2.1. RQ1: Performance on vulnerability presence detection

In this experiment, we evaluate the performance of TrVD in detecting the existence of vulnerability. State-of-the-art DL-based vulnerability detection methods, including VulDeePecker, SySeVR, VulBERTa, Devign and Reveal are used for performance comparison with TrVD. A brief introduction of the baseline methods is provided as follows:

- **VulDeePecker** (Li et al., 2018) and **SySeVR** (Li, Zou, Xu, Jin et al., 2022) are two slice-based methods, which feed code gadgets, i.e., groups of code statements that have control or dependency relationships with respect to certain interesting code elements such as library/API calls and array usages, to recurrent neural networks for vulnerability detection.
- **VulBERTa** (Hanif & Maffeis, 2022) adopts BERT as strategy, by pre-training a RoBERTa model with MLM (Masked Language Modelling) on a large corpus of C/C++ programs, and then fine-tuning on certain vulnerability datasets to train the vulnerability detectors.
- **Devign** (Zhou et al., 2019) and **Reveal** (Chakraborty et al., 2021) are two graph-based approaches that leverage GGNN (Gated Graph Neural Network) to extract vulnerability indicative features from the CPG (Code Property Graph) (Yamaguchi et al., 2014) representation of the code to be detected.
- Two widely-used models, denoted as Baseline-BiLSTM and Baseline-TextCNN, which take in the lexically parsed token sequence and detect vulnerability with BiLSTM and TextCNN respectively, also serve as baselines.

Table 3
Comparison with SOTA methods on vulnerability presence detection.

| Dataset | Method | Acc. | F1 | Prec. | Rec. |
|--------------|------------------|--------------|--------------|--------------|--------------|
| Ours | Baseline-BiLSTM | 83.12 | 82.68 | 81.14 | 84.28 |
| | Baseline-TextCNN | 86.00 | 85.64 | 86.49 | 84.80 |
| | VulDeePecker | 87.49 | 87.38 | 86.74 | 88.02 |
| | SySeVR | 87.78 | 87.72 | 86.67 | 88.79 |
| | Devign | 87.21 | 87.16 | 86.62 | 87.50 |
| | TrVD | 89.29 | 88.19 | 92.85 | 83.97 |
| D2A | Baseline-BiLSTM | 56.42 | – | – | – |
| | Baseline-TextCNN | 55.18 | – | – | – |
| | C-BERT | 60.20 | – | – | – |
| | Devign | 60.34 | – | – | – |
| | VulBERTa | 62.30 | – | – | – |
| | TrVD | 64.15 | – | – | – |
| VulDeePecker | Baseline-BiLSTM | – | 66.97 | 52.58 | – |
| | Baseline-TextCNN | – | 75.80 | 63.48 | – |
| | VulDeePecker | – | 92.90 | 91.90 | – |
| | Devign | – | 92.52 | 93.76 | – |
| | VulBERTa | – | 93.03 | 95.76 | – |
| | TrVD | – | 92.18 | 96.83 | – |
| Reveal | Baseline-BiLSTM | 77.13 | 39.11 | 26.76 | 72.61 |
| | Baseline-TextCNN | 73.22 | 37.42 | 24.50 | 79.13 |
| | REVEAL | 84.37 | 41.25 | 30.91 | 60.91 |
| | Devign | 80.71 | 39.55 | 28.33 | 65.47 |
| | VulBERTa | 84.48 | 45.27 | 35.18 | 63.48 |
| | TrVD | 86.54 | 47.79 | 38.24 | 63.69 |

**Answer to RQ1:** Among the DL-based methods evaluated, TrVD shows competitive performances, especially on the accuracy, precision and F1 metrics. Also, reliable detection on the real-world programs is still challenging, improvements should be made by the DL-based methods to achieve a more satisfactory detection performance.

Table 4
Quantity distribution of vulnerable functions per vulnerability type.

| Label | Type | #Sample | Label | Type | #Sample | Label | Type | #Sample | Label | Type | #Sample |
|-------|---------|---------|-------|---------|---------|-------|-----------------|---------|-------|---------|---------|
| 1 | CWE-22 | 7600 | 23 | CWE-396 | 54 | 45 | CWE-672 | 30 | 67 | CWE-222 | 18 |
| 2 | CWE-73 | 978 | 24 | CWE-398 | 177 | 46 | CWE-675 | 300 | 68 | CWE-223 | 18 |
| 3 | CWE-77 | 8200 | 25 | CWE-400 | 1212 | 47 | CWE-681 | 5368 | 69 | CWE-247 | 18 |
| 4 | CWE-119 | 27571 | 26 | CWE-404 | 638 | 48 | CWE-690 | 1592 | 70 | CWE-273 | 36 |
| 5 | CWE-134 | 6175 | 27 | CWE-426 | 316 | 49 | CWE-758 | 581 | 71 | CWE-338 | 17 |
| 6 | CWE-176 | 82 | 28 | CWE-427 | 805 | 50 | CWE-763 | 8589 | 72 | CWE-397 | 6 |
| 7 | CWE-190 | 6393 | 29 | CWE-459 | 108 | 51 | CWE-771 | 244 | 73 | CWE-475 | 36 |
| 8 | CWE-191 | 4741 | 30 | CWE-464 | 82 | 52 | CWE-772 | 2299 | 74 | CWE-478 | 18 |
| 9 | CWE-200 | 36 | 31 | CWE-467 | 54 | 53 | CWE-789 | 1516 | 75 | CWE-479 | 18 |
| 10 | CWE-252 | 617 | 32 | CWE-468 | 37 | 54 | CWE-843 | 126 | 76 | CWE-483 | 20 |
| 11 | CWE-253 | 684 | 33 | CWE-469 | 36 | 55 | CWE-908 | 1039 | 77 | CWE-484 | 18 |
| 12 | CWE-271 | 252 | 34 | CWE-476 | 484 | 56 | CWE-912 | 300 | 78 | CWE-535 | 36 |
| 13 | CWE-284 | 216 | 35 | CWE-480 | 54 | 57 | CWE-943 | 820 | 79 | CWE-570 | 16 |
| 14 | CWE-319 | 328 | 36 | CWE-534 | 36 | 58 | CWE-1078 | 59 | 80 | CWE-571 | 16 |
| 15 | CWE-325 | 72 | 37 | CWE-563 | 606 | 59 | CWE-1105 | 36 | 81 | CWE-587 | 18 |
| 16 | CWE-327 | 72 | 38 | CWE-588 | 121 | 60 | CWE-1177 | 36 | 82 | CWE-605 | 18 |
| 17 | CWE-328 | 54 | 39 | CWE-606 | 820 | 61 | CWE-119,672 | 497 | 83 | CWE-620 | 18 |
| 18 | CWE-362 | 90 | 40 | CWE-617 | 469 | 62 | CWE-119,672,415 | 161 | 84 | CWE-785 | 18 |
| 19 | CWE-369 | 1387 | 41 | CWE-628 | 36 | 63 | CWE-1390,344 | 319 | 85 | CWE-835 | 6 |
| 20 | CWE-377 | 144 | 42 | CWE-665 | 311 | 64 | CWE-1390,522 | 64 | | | |
| 21 | CWE-390 | 90 | 43 | CWE-666 | 90 | 65 | CWE-15,642 | 82 | | | |
| 22 | CWE-391 | 54 | 44 | CWE-667 | 200 | 66 | CWE-459,212 | 72 | | | |

Table 5
Comparison with SOTA methods on vulnerability type detection.

| Dataset | Method | Acc | Weighted F1 | Weighted Prec | Weighted Rec |
|--------------------|------------------|--------------|----------------|---------------|--------------|
| Ours | Baseline-BiLSTM | 74.54 | 68.38 | 64.55 | 74.53 |
| | Baseline-TextCNN | 76.21 | 70.81 | 70.05 | 76.21 |
| | VulDeePecker | 83.31 | 81.93 | 81.86 | 83.31 |
| | SySeVR | 84.88 | 84.21 | 84.86 | 84.89 |
| | Devign | 83.47 | 82.49 | 84.27 | 83.47 |
| | TrVD | 85.85 | 85.54 | 85.98 | 85.83 |
| muVulDeePecker | Baseline-BiLSTM | – | 65.93 | – | – |
| | Baseline-TextCNN | – | 65.95 | – | – |
| | VulDeePecker | – | 93.61 | – | – |
| | muVulDeePecker | – | 94.69 | – | – |
| | VulBERTa | – | 99.59 | – | – |
| | TrVD | – | 94.93 | – | – |

#### 5.2.2. RQ2: Performance on vulnerability type detection

In this experiment, we evaluate the performance of TrVD in pinpointing the specific vulnerability types. Our constructed dataset, which consists of 98,181 vulnerable functions that are well-labeled to 118 different CWE-IDs, is used for the evaluation, considering the adequacy of the vulnerable samples and the richness of the vulnerability types it contains. As the CWE-IDs are organized hierarchically (i.e., low-level CWE-IDs are affiliated to some higher-level CWE-IDs, e.g., CWE-119 includes CWE-787 and CWE-788 as sub-types), we adopt the same policy as in muVulDeePecker (Zou et al., 2021) to aggregate the CWE-IDs to the third level of the CWE-ID tree in its research concept view, and regard these third-level CWE-IDs as the vulnerability types. For example, for a vulnerable sample originally marked as CWE-787 or the lower-level type CWE-121, we reassign its vulnerability type to the CWE-ID at the third level of the CWE-ID tree, which is CWE-119 in this case. Also, there are CWE-IDs that are children of multiple third-level CWE-IDs. For example, the low-level CWE-416 can be attributed to either the CWE-119 or CWE-672 at the third level of the CWE-ID tree. For the presentation consistency, we use their joint (i.e., "CWE-119, CWE-672") to denote the vulnerability type for CWE-416. Finally, this leads to 85 different types as listed in Table 4.

It should be noted that, muVulDeePecker (Zou et al., 2021) also publishes its dataset for vulnerability type detection. However, as the source code of muVulDeePecker is not made public, we only evaluate and compare the performance of TrVD on its dataset, which involves 40 vulnerability types.

**Answer to RQ2:** TrVD is effective for the multi-class vulnerability type detection task. It shows promising and competitive performance among the DL-based baseline methods.

#### 5.2.3. RQ3: Ablation study on TrVD's core components

As illustrated by the evaluation results in the above sections, as a whole, TrVD outperforms SOTA methods in detecting either the presence or the specific types of the given vulnerabilities. We attribute such superior detection performance to the elaborate designs in TrVD's core components. Hereby, we conduct the ablation study as follows to better understand the roles that TrVD's different components play in contributing to the overall detection performance.

##### (1) The effects of the AST Decomposer

As discussed in Sections 2.3 and 4.2, we design the decomposer to convert a full AST into a sequence of sub-trees of restricted sizes, with the purpose of alleviating the cost of subsequent neural encoders in handling large/deep ASTs, and better disclosing the subtle vulnerability semantics conveyed in the small-sized code unit. To evaluate its effects, we substitute our decomposer with the following alternatives and compare their detection performances on both vulnerability presence detection and vulnerability type detection tasks. Specifically, these alternatives include:

- AST_{TBCNN} , where neither decomposition nor serialization is enforced, but the whole AST is directly fed into the tree-based convolutional neural network (TBCNN) (Mou et al., 2016) for feature extraction;
- AST_{TreeLSTM} , where the whole AST is directly fed into the tree-structured long short-term memory network (Tree-LSTM) (Tai et al., 2015) for feature extraction;

Table 6
Effects of the AST Decomposer.

| (a) Performance on Vulnerability Presence Detection | | | | | | | | | | |
|-----------------------------------------------------|---------|--------------|-------------------------|---------------|--------------|---------------------|--------------|-------------------------|---------------|--------------|
| Method | Dataset | Acc | F1 | Prec | Rec | Dataset | Acc | F1 | Prec | Rec |
| AST _{TBCNN} | Ours | 87.71 | 87.62 | 86.88 | 88.37 | VulDeePecker | 90.01 | 91.08 | 94.08 | 88.26 |
| AST _{TreeLSTM} | | 86.81 | 86.55 | 86.78 | 86.33 | | 90.23 | 91.88 | 94.59 | 89.33 |
| AST _{AttRvNN} | | 87.36 | 87.25 | 86.83 | 87.61 | | 91.68 | 91.40 | 94.62 | 88.40 |
| AST _{RvNN} | | 87.01 | 86.75 | 87.03 | 86.48 | | 91.17 | 90.92 | 93.57 | 88.40 |
| TrVD | | 89.29 | 88.19 | 92.85 | 83.97 | | 92.52 | 92.18 | 96.83 | 87.95 |
| AST _{TBCNN} | D2A | 58.89 | 61.90 | 59.40 | 64.61 | Reveal | 83.45 | 42.53 | 30.88 | 68.28 |
| AST _{TreeLSTM} | | 58.72 | 60.03 | 58.11 | 62.08 | | 83.19 | 42.24 | 30.52 | 68.57 |
| AST _{AttRvNN} | | 59.08 | 62.21 | 58.95 | 65.86 | | 83.98 | 43.05 | 32.17 | 65.03 |
| AST _{RvNN} | | 58.55 | 61.16 | 58.73 | 63.79 | | 83.66 | 42.59 | 31.30 | 66.61 |
| TrVD | | 64.15 | 63.14 | 67.34 | 59.43 | | 86.54 | 47.79 | 38.24 | 63.69 |
| (b) Performance on Vulnerability Type Detection | | | | | | | | | | |
| Method | Dataset | Acc | Weighted F1 | Weighted Prec | Weighted Rec | Dataset | Acc | Weighted F1 | Weighted Prec | Weighted Rec |
| AST _{TBCNN} | Ours | 82.54 | 84.78 | 85.27 | 84.55 | mu-VulDeePecker | 89.42 | 88.72 | 88.94 | 89.42 |
| AST _{TreeLSTM} | | 81.89 | 83.64 | 84.47 | 83.90 | | 88.11 | 87.44 | 87.47 | 88.11 |
| AST _{AttRvNN} | | 83.94 | 84.64 | 85.05 | 83.90 | | 91.95 | 91.61 | 91.62 | 91.95 |
| AST _{RvNN} | | 82.47 | 84.55 | 85.55 | 84.47 | | 90.74 | 90.23 | 90.40 | 90.74 |
| TrVD | | 85.85 | 85.54 | 85.98 | 85.83 | | 95.01 | 94.93 | 95.12 | 95.01 |

- ASTNN (Zhang et al., 2019), where the AST is also decomposed into a set of sub-trees, but differently, compound statements are always split as described in Section 4.2.

##### (2) The effects of the Semantics Collector

As discussed in Section 4.3, we design an attention-augmented tree-structured RvNN to collect the semantics from a sub-tree. To evaluate its effects, we substitute it with other well-adopted tree-oriented neural networks and compare their performances, including:

- TrVD_{TBCNN}, where the TBCNN is used to encode each sub-tree;
- TrVD_{TreeLSTM}, where the Tree-LSTM is used to encode each sub-tree;
- TrVD_{RvNN}, where the naive RvNN without attention mechanism is used to encode each sub-tree.

Table 7 reports the experimental results. As the values show, TrVD basically performs better than its alternatives, validating the superiority of our attention-augmented RvNN encoder against other tree-oriented neural encoders in collecting the latent code semantics. In addition, these less significant performance improvements than the ones observed in the last experiment implies that, the Decomposer plays a more impactful role than the attention-augmented RvNN semantic Collector in contributing to the competitive detection performance of TrVD.

##### (3) The effects of the Semantics Focuser

In Section 4.4, we resort to a Transformer-style neural network to aggregate the semantics collected from the decomposed sub-trees into a dense vulnerability-specific vector. To evaluate its effects, we substitute it with other aggregation strategies and compare their performances in this set of experiments. The specific replacement strategies include TrVD_{AvgPooling}, TrVD_{MaxPooling}, TrVD_{TextCNN}, TrVD_{DPCNN} and TrVD_{BiLSTM}, where the average pooling, max pooling, TextCNN (Kim, 2014), DPCNN (Johnson & Zhang, 2017) and BiLSTM are used respectively to aggregate the semantic vectors collected from all the sub-trees of an AST.

**Answer to RQ3:** The core designs in TrVD's Decomposer, Collector and Focuser all play active roles, with the well-designed tree decomposition algorithm in the Decomposer contributing the most to the competitive detection performance of TrVD.

#### 5.2.4. RQ4: Efficiency analysis

In this section, we analyze the runtime overhead of TrVD. Given a code snippet, it takes TrVD two major steps, the data preparation step and the neural classification step, to accomplish its detection. The data preparation step, supported by the *AST Generator* and *AST Decomposer* in TrVD, extracts the AST from the input code snippet and decomposes it into a sequence of sub-trees. The neural classification step, supported by the *Semantics Collector*, *Suspicious Semantics Focuser* and *Vulnerability Detector* in TrVD, makes a prediction with the trained neural network model.

Specifically, we feed each sample from our dataset into TrVD for detection, and meanwhile record the time consumed in the two steps

Table 7
Effects of the Semantics Collector.

| (a) Performance on Vulnerability Presence Detection | | | | | | | | | | |
|-----------------------------------------------------|---------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Method | Dataset | Acc | F1 | Prec | Rec | Dataset | Acc | F1 | Prec | Rec |
| TrVD _{TBCNN} | Ours | 88.73 | 84.87 | 85.68 | 84.07 | VulDeePecker | 92.31 | 92.15 | 94.39 | 89.98 |
| TrVD _{TreeLSTM} | | 89.09 | 85.13 | 86.38 | 83.91 | | 92.01 | 91.76 | 94.83 | 88.88 |
| TrVD _{RvNN} | | 89.11 | 85.35 | 88.22 | 82.66 | | 92.41 | 92.12 | 96.09 | 88.47 |
| TrVD | | 89.29 | 88.19 | 92.85 | 83.97 | | 92.52 | 92.18 | 96.83 | 87.95 |
| TrVD _{TBCNN} | D2A | 62.45 | 61.99 | 62.46 | 61.53 | Reveal | 85.81 | 46.66 | 36.48 | 64.72 |
| TrVD _{TreeLSTM} | | 63.05 | 62.17 | 65.30 | 59.33 | | 86.32 | 47.30 | 37.53 | 67.93 |
| TrVD _{RvNN} | | 62.42 | 61.78 | 65.11 | 58.77 | | 86.03 | 46.69 | 36.76 | 63.98 |
| TrVD | | 64.15 | 63.14 | 67.34 | 59.43 | | 86.54 | 47.79 | 38.24 | 63.69 |

| (b) Performance on Vulnerability Type Detection | | | | | | | | | | |
|-------------------------------------------------|---------|--------------|----------------|---------------|--------------|---------|--------------|----------------|---------------|--------------|
| Method | Dataset | Acc | Weighted F1 | Weighted Prec | Weighted Rec | Dataset | Acc | Weighted F1 | Weighted Prec | Weighted Rec |
| TrVD _{TBCNN} | Ours | 85.39 | 84.94 | 85.38 | 85.39 | mu- | 93.55 | 94.18 | 94.37 | 94.25 |
| TrVD _{TreeLSTM} | | 84.71 | 85.34 | 85.93 | 85.71 | | 94.25 | 94.90 | 95.04 | 94.98 |
| TrVD _{RvNN} | | 85.67 | 85.32 | 85.78 | 85.68 | | 94.98 | 93.43 | 93.84 | 93.55 |
| TrVD | | 85.85 | 85.54 | 85.98 | 85.83 | | 95.01 | 94.93 | 95.12 | 95.01 |

Table 8
Effects of the Semantics Focuser.

| (a) Performance on Vulnerability Presence Detection | | | | | | | | | | |
|-----------------------------------------------------|---------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Method | Dataset | Acc | F1 | Prec | Rec | Dataset | Acc | F1 | Prec | Rec |
| TrVD _{AvgPooling} | Ours | 88.31 | 87.66 | 86.97 | 88.37 | VulDeePecker | 91.74 | 91.48 | 94.40 | 88.74 |
| TrVD _{MaxPooling} | | 88.86 | 87.62 | 90.85 | 84.61 | | 91.63 | 91.32 | 94.88 | 88.02 |
| TrVD _{TextCNN} | | 88.46 | 86.98 | 90.63 | 83.62 | | 91.84 | 91.57 | 94.70 | 88.64 |
| TrVD _{DPCNN} | | 89.12 | 87.33 | 92.87 | 82.42 | | 91.79 | 91.50 | 94.80 | 88.43 |
| TrVD _{BiLSTM} | | 88.73 | 87.07 | 88.83 | 85.37 | | 91.93 | 91.69 | 94.94 | 88.65 |
| TrVD | | 89.29 | 88.19 | 92.85 | 83.97 | | 92.52 | 92.18 | 96.83 | 87.95 |
| TrVD _{AvgPooling} | D2A | 61.96 | 62.76 | 61.09 | 64.53 | Reveal | 85.66 | 46.93 | 36.82 | 64.71 |
| TrVD _{MaxPooling} | | 62.50 | 62.54 | 61.92 | 63.17 | | 85.82 | 46.97 | 36.80 | 64.92 |
| TrVD _{TextCNN} | | 62.32 | 61.83 | 62.34 | 61.32 | | 85.97 | 47.01 | 37.06 | 64.27 |
| TrVD _{DPCNN} | | 63.49 | 62.16 | 64.76 | 59.76 | | 86.29 | 47.37 | 37.54 | 64.18 |
| TrVD _{BiLSTM} | | 63.23 | 61.61 | 63.82 | 60.15 | | 86.12 | 47.31 | 37.44 | 64.25 |
| TrVD | | 64.15 | 63.14 | 67.34 | 59.43 | | 86.54 | 47.79 | 38.24 | 63.69 |

| (b) Performance on Vulnerability Type Detection | | | | | | | | | | |
|-------------------------------------------------|---------|--------------|----------------|---------------|--------------|---------|--------------|----------------|---------------|--------------|
| Method | Dataset | Acc | Weighted F1 | Weighted Prec | Weighted Rec | Dataset | Acc | Weighted F1 | Weighted Prec | Weighted Rec |
| TrVD _{AvgPooling} | Ours | 73.34 | 66.74 | 62.53 | 73.34 | mu- | 92.06 | 91.70 | 91.73 | 92.06 |
| TrVD _{MaxPooling} | | 75.71 | 70.01 | 67.79 | 75.71 | | 92.35 | 92.11 | 92.40 | 92.35 |
| TrVD _{TextCNN} | | 84.41 | 83.57 | 83.99 | 84.41 | | 94.72 | 94.62 | 94.73 | 94.72 |
| TrVD _{DPCNN} | | 84.88 | 84.20 | 84.86 | 84.88 | | 93.17 | 93.11 | 93.36 | 93.17 |
| TrVD _{BiLSTM} | | 85.06 | 84.59 | 85.26 | 85.06 | | 94.96 | 94.87 | 95.03 | 94.96 |
| TrVD | | 85.85 | 85.54 | 85.98 | 85.83 | | 95.01 | 94.93 | 95.12 | 95.01 |

Fig. 4. Evaluation on the runtime overhead of TrVD's two major steps.

respectively. As depicted in Fig. 4 about the runtime overheads, TrVD consumes averagely 9.8 ms in obtaining the sequence of decomposed sub-trees, and 53.41 ms in making a prediction with the trained model.

To give a more intuitive sense of the efficiency of TrVD, we also compare its runtime overhead with five other DL-based methods, including Baseline-BiLSTM, Baseline-TextCNN, VulDeePecker, SySeVR and Devign. Their average runtime overheads are reported in Fig. 5.

As it shows, Baseline-TextCNN and Baseline-BiLSTM are the most efficient methods, since they only apply simple lexical analysis to obtain the source code tokens and use a CNN and BiLSTM model to detect vulnerability respectively. On average, it takes them only 25 ms and 38 ms respectively to finish the analysis of a function in our dataset. The graph-based method Devign is the most computationally intensive, due to its heavy weight code property graph construction phase and the GGNN based feature extraction phase. Averagely, it takes 1154 ms to analyze a function, which is about 46 times slower than Baseline-TextCNN. As for VulDeePecker and SySeVR, code slicing using control or data dependency is enforced to obtain code gadgets from the PDG, which makes their data preparation phase heavy-weight. Accordingly, VulDeePecker and SySeVR take averagely 876 ms and 993 ms to detect a function, while TrVD takes 63 ms on average to analyze a function, which, though not as efficient as the token-based methods, is about 14–16 times faster than VulDeePecker and SySeVR.

| Method | Runtime Overhead (seconds) |
|------------------|----------------------------|
| Baseline-BiLSTM | 0.038 |
| Baseline-TextCNN | 0.025 |
| VulDeePecker | 0.876 |
| SySeVR | 0.993 |
| Devign | 1.154 |
| TrVD | 0.063 |

**Answer to RQ4:** By following a light-weight data preparation strategy, TrVD is significantly learning-effective yet cost-efficient among the DL-based baseline methods.

## 6. Discussion

### 6.1. Applicability issues

As discussed in Section 2.2, TrVD opts to operate on the easy-to-obtain AST to ensure its applicability. Other purposely-crafted code representations, such as PDG and code gadgets, are difficult to precisely construct for non-compilable code fragments or simply cannot be extracted without the presence of certain code elements in the code (thus severely diminish the applicability of the detection methods that rely on these representations). By contrast, the AST can always be statically constructed for almost any code fragment. With the tree-sitter parsing library, TrVD succeeds in analyzing each single sample from all five datasets we evaluated on. Taking our constructed dataset as an example, TrVD is able to process all the functions, while the baseline methods SySeVR and VulDeePecker only allow 60.5% of the whole samples to be successfully analyzed. Also, both our dataset and VulDeePecker's dataset are mainly constructed from the SARD dataset, but the number of samples in their dataset (i.e., the ones that can be processed by their method) is far less, which is 68.6% of ours.

In this work, TrVD is evaluated on the code written in C and C++, which are among the programming languages that have been hit hard by vulnerabilities. But theoretically, it can be extended to the vulnerability detection for code written in other programming languages in a straightforward way, as long as the AST can be constructed from the target code snippets. The tree-sitter library that TrVD relies upon to construct ASTs, currently provides fairly mature parsing ability for over 40 different kinds of programming languages. In our future work, we plan to extend TrVD to support the vulnerability detection for more programming languages, such as the widely-used Java and the emerging Solidity used for smart contract development. Definitely, the tree decomposition algorithm as well as the neural encoding and classification models can be adjusted to accommodate to the specific characteristics of the language to be analyzed, so as to ensure an optimal detection performance.

As a learning-based detection method, TrVD still suffers from the limitation of only reporting black-box detection results, i.e., a vulnerable/non-vulnerable prediction or the specific vulnerability type without explaining, which is different from the rule-based methods by providing detection results with additional information to imply possible bug-triggering paths of the detected vulnerabilities (Cheng, Nie et al., 2022). Driven by this purpose, several works have emerged recently with the goal of better explaining the detection results, by highlighting the statements (Ding et al., 2022; Li, Wang et al., 2021; Li, Zou, Xu, Chen et al., 2022; Nguyen et al., 2021) or paths (Cheng, Zhang et al., 2022) with significant contributions to the prediction outcomes using either explainable AI techniques, such as GNN-explainer (Li, Wang et al., 2021) and mutual information maximization (Nguyen et al., 2021), or jointly-trained multi-grained models (Cheng, Zhang et al., 2022; Ding et al., 2022). TrVD can be also adjusted to highlight the indicative subtrees (i.e., statements) with the attention scores or other explainable AI techniques (e.g., local approximation methods). We leave this as an interesting future work.

### 6.2. Dataset issues

As pointed out by existing works (Cheng, Zhang et al., 2022; Croft, Babar, & Kholoosi, 2023; Croft, Xie, & Babar, 2023; Hanif & Maffeis, 2022; Jimenez et al., 2019; Tian, Tian, Lv, & Chen, 2023), the sample mislabeling in the vulnerability datasets remains a common and unresolved problem. But the mislabeling ratio in the datasets we used is relatively low, as the included samples are labeled either by security experts or through a well-designed differential analysis technique (Zheng et al., 2021). Thus, as a DL-based method, TrVD should be resilient to the occasional label noises during model training. The influence of the low-ratio noises on the testing performance should also be minor.

Those real-world vulnerability datasets with satisfying annotations are critically lacking. The datasets we used are primarily labeled at the same granularity of isolated functions as the one used by the majority of existing works. In practice, vulnerabilities can happen across function boundaries. It is thus not always correct to mark a function as vulnerable simply because the vulnerability is disclosed in this function, while additional inter-procedural information, such as the macros and the functions that it calls, should be also taken into consideration to form the complete bug-triggering contexts. Unfortunately, constructing such in-scale datasets that are comprised of precisely-labeled bug-triggering code contexts is non-trivial, which requires continuous laborious efforts from domain experts. But even so, we argue that TrVD should work on such data with moderate adjustments, since all its components are not restricted to be applied on isolated functions.

The numbers of samples that reside in different classes are imbalanced in some of our datasets. For example, the ratio between the vulnerable and non-vulnerable samples in the Reveal dataset is about 1:10. During the model training, we under-sample the non-vulnerable samples to obtain balanced training sets. Other data imbalance handling techniques, such as SMOTE re-sampling, may also be feasible. Besides, the emerging bug and vulnerability injection techniques (Ding et al., 2022; Nong, Ou, Pradel, Chen, & Cai, 2022; Yu, Yuzhe, Michael, Feng, & Haipeng, 2023) deserve further investigation and adaptation to enrich the diversity and the size of vulnerable samples, which help solve the data-imbalance issue and may improve the detection performance of DL-based methods. We leave them as future works.

## 7. Conclusion

In this work, we present TrVD, a novel source code vulnerability detection approach that follows the DL-based paradigm. Through the comprehensive contributions from our elaborate designs, including the novel tree decomposition algorithm, the attention-augmented sub-tree encoder and the vulnerability-indicative semantic focuser, TrVD turns out to be accurate, efficient and practically applicable in detecting either the presence of vulnerability or the specific vulnerability type. The extensive experiments conducted on five large datasets consisting of both synthetic and real-world samples demonstrate the superiority of TrVD against SOTA DL-based methods in the aspects of detection performance and runtime overhead. The effectiveness of TrVD's core designs are also well-studied with a thorough ablation study. As our future works, we plan to extend TrVD to investigate advanced embedding models (such as large-scale pre-trained models, e.g., CodeBert) for token representation learning, more diverse programming languages, and interpretations for the detection results.
