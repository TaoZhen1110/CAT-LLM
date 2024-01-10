CAT-LLM: Prompting Large Language Models with Text Style Definition for Chinese Article-style Transfer
====
Project Structure<br>
----
The project is organized into several key directories and modules. Here's an overview of the project structure:<br>

├── bert-base-chinese                         # Store bert-base-chines file used in our experiment, .<br>
├── data                                      # Store five dataset.<br>
├── Models                                    # Core codebase.<br>
│   ├── Baichuan                              # Three Baichuan operations.<br>
│   ├── ChatGLM                               # Three ChatGLM operations.<br>
│   └── GPT-3.5                               # Three GPT-3.5 operations.<br>
├── sentence_word_define_dataset              # Store sentence_word_define_dataset.<br>
├── TST_sentence                              # Store TST_sentence classification models.<br>
├── ACC_BLEU_BERT                             # Store BLEU_BERT metrics.<br>
├── All_style_define                          # Store TSD module.<br>
└── Content_preserve                          # Store Content_preserve code.<br>
