# UUParser: A transition-based dependency parser for Universal Dependencies

This parser is based on [Eli Kiperwasser's transition-based parser](http://github.com/elikip/bist-parser) using BiLSTM feature extractors.
We adapted the parser to Universal Dependencies and extended it as described in these papers:

* (Version 1.0) Adaptation to UD + removed POS tags from the input + added character vectors + use pseudo-projective:
>Miryam de Lhoneux, Yan Shao, Ali Basirat, Eliyahu Kiperwasser, Sara Stymne, Yoav Goldberg, and Joakim Nivre. 2017. From Raw Text to Universal Dependencies - Look, No Tags! In Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies.

* (Version 2.0) Removed the need for pseudo-projective parsing by using a swap transition and creating a partially dynamic oracle as described in:
>Miryam de Lhoneux, Sara Stymne and Joakim Nivre. 2017. Arc-Hybrid Non-Projective Dependency Parsing with a Static-Dynamic Oracle. In Proceedings of the The 15th International Conference on Parsing Technologies (IWPT).

* (Version 2.3) Added POS tags back in, extended cross-treebank functionality and use of external embeddings and some tuning of default hyperparameters:

>Aaron Smith, Bernd Bohnet, Miryam de Lhoneux, Joakim Nivre, Yan Shao and Sara Stymne. 2018. 82 Treebanks, 34 Models: Universal Dependency Parsing with Cross-Treebank Models. In Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies.

The techniques behind the original parser are described in the paper [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198).

#### Citation

If you make use of this software for research purposes, we'll appreciate if you cite the following:

If you use version 2.0 or later:

    @InProceedings{delhoneux17arc,
        author    = {Miryam de Lhoneux and Sara Stymne and Joakim Nivre},
        title     = {Arc-Hybrid Non-Projective Dependency Parsing with a Static-Dynamic Oracle},
        booktitle = {Proceedings of the The 15th International Conference on Parsing Technologies (IWPT).},
        year      = {2017},
        address = {Pisa, Italy}
    }

If you use version 1.0:

    @InProceedings{uu-conll17,
        author    = {Miryam de Lhoneux and Yan Shao and Ali Basirat and Eliyahu Kiperwasser and Sara Stymne and Yoav Goldberg and Joakim Nivre},
        title     = {From Raw Text to Universal Dependencies -- Look, No Tags!},
        booktitle = {Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies. },
        year      = {2017},
        address = {Vancouver, Canada}
    }

And the original parser paper:

    @article{DBLP:journals/tacl/KiperwasserG16,
        author    = {Eliyahu Kiperwasser and Yoav Goldberg},
        title     = {Simple and Accurate Dependency Parsing Using Bidirectional {LSTM}
               Feature Representations},
        journal   = {{TACL}},
        volume    = {4},
        pages     = {313--327},
        year      = {2016},
        url       = {https://transacl.org/ojs/index.php/tacl/article/view/885},
        timestamp = {Tue, 09 Aug 2016 14:51:09 +0200},
        biburl    = {http://dblp.uni-trier.de/rec/bib/journals/tacl/KiperwasserG16},
        bibsource = {dblp computer science bibliography, http://dblp.org}
    }

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact miryam.de\_lhoneux@lingfil.uu.se
